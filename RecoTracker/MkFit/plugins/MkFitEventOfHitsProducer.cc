#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitEventOfHits.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

// mkFit includes
#include "mkFit/HitStructures.h"
#include "mkFit/MkStdSeqs.h"
#include "LayerNumberConverter.h"

class MkFitEventOfHitsProducer : public edm::global::EDProducer<> {
public:
  explicit MkFitEventOfHitsProducer(edm::ParameterSet const& iConfig);
  ~MkFitEventOfHitsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  void fill(const std::vector<const TrackingRecHit*>& hits,
            mkfit::EventOfHits& eventOfHits,
            const MkFitGeometry& mkFitGeom) const;

  const edm::EDGetTokenT<MkFitHitWrapper> pixelHitsToken_;
  const edm::EDGetTokenT<MkFitHitWrapper> stripHitsToken_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> pixelClusterIndexToHitToken_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> stripClusterIndexToHitToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualityToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::EDPutTokenT<MkFitEventOfHits> putToken_;
  const bool useStripStripQualityDB_;
};

MkFitEventOfHitsProducer::MkFitEventOfHitsProducer(edm::ParameterSet const& iConfig)
    : pixelHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelHits"))},
      stripHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("stripHits"))},
      pixelClusterIndexToHitToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelHits"))},
      stripClusterIndexToHitToken_{consumes(iConfig.getParameter<edm::InputTag>("stripHits"))},
      mkFitGeomToken_{esConsumes()},
      putToken_{produces<MkFitEventOfHits>()},
      useStripStripQualityDB_{iConfig.getParameter<bool>("useStripStripQualityDB")} {
  if (useStripStripQualityDB_) {
    qualityToken_ = esConsumes();
    geomToken_ = esConsumes();
  }
}

void MkFitEventOfHitsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("pixelHits", edm::InputTag{"mkFitSiPixelHits"});
  desc.add("stripHits", edm::InputTag{"mkFitSiStripHits"});
  desc.add("useStripStripQualityDB", true)->setComment("Use SiStrip quality DB information");

  descriptions.addWithDefaultLabel(desc);
}

void MkFitEventOfHitsProducer::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& pixelHits = iEvent.get(pixelHitsToken_);
  const auto& stripHits = iEvent.get(stripHitsToken_);
  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);

  auto eventOfHits = std::make_unique<mkfit::EventOfHits>(mkFitGeom.trackerInfo());
  mkfit::StdSeq::Cmssw_LoadHits_Begin(*eventOfHits, {&pixelHits.hits(), &stripHits.hits()});

  if (useStripStripQualityDB_) {
    std::vector<mkfit::DeadVec> deadvectors(mkFitGeom.layerNumberConverter().nLayers());
    const auto& siStripQuality = iSetup.getData(qualityToken_);
    const auto& trackerGeom = iSetup.getData(geomToken_);
    const auto& badStrips = siStripQuality.getBadComponentList();
    for (const auto& bs : badStrips) {
      const DetId detid(bs.detid);
      const auto& surf = trackerGeom.idToDet(detid)->surface();
      bool isBarrel = (mkFitGeom.topology()->side(detid) == static_cast<unsigned>(TrackerDetSide::Barrel));
      const auto ilay = mkFitGeom.mkFitLayerNumber(detid);
      const auto q1 = isBarrel ? surf.zSpan().first : surf.rSpan().first;
      const auto q2 = isBarrel ? surf.zSpan().second : surf.rSpan().second;
      if (bs.BadModule)
        deadvectors[ilay].push_back({surf.phiSpan().first, surf.phiSpan().second, q1, q2});
      else { //assume that BadApvs are filled in sync with BadFibers
        auto const& topo = dynamic_cast<const StripTopology&>(trackerGeom.idToDet(detid)->topology());
        int firstApv = -1;
        int lastApv = -1;

        auto addRangeAPV = [&topo, &surf, &q1, &q2](int first, int last, mkfit::DeadVec& dv) {
          auto const firstPoint = surf.toGlobal(topo.localPosition(first * 128));
          auto const lastPoint = surf.toGlobal(topo.localPosition((last + 1) * 128));
          float phi1 = firstPoint.phi();
          float phi2 = lastPoint.phi();
          if (reco::deltaPhi(phi1, phi2) > 0)
	    std::swap(phi1, phi2);
          LogTrace("SiStripBadComponents")<<"insert bad range "<<first<<" to "<<last<<" "<<phi1<<" "<<phi2;
          dv.push_back({phi1, phi2, q1, q2});
        };

        for (int apv = 0; apv < 6; ++apv) {
          const bool isBad = bs.BadApvs & (1 << apv);
          if (isBad)
            LogTrace("SiStripBadComponents")<<"bad apv "<<apv<<" on "<<bs.detid;
          if (isBad) {
            if (lastApv == -1) {
              firstApv = apv;
              lastApv = apv;
            } else if (lastApv + 1 == apv)
              lastApv++;

            if (apv == 5)
              addRangeAPV(firstApv, lastApv, deadvectors[ilay]);
          } else if (firstApv != -1) {
            addRangeAPV(firstApv, lastApv, deadvectors[ilay]);
            //and reset
            firstApv = -1;
            lastApv = -1;
          }
        }
      }
    }
    mkfit::StdSeq::LoadDeads(*eventOfHits, deadvectors);
  }

  fill(iEvent.get(pixelClusterIndexToHitToken_).hits(), *eventOfHits, mkFitGeom);
  fill(iEvent.get(stripClusterIndexToHitToken_).hits(), *eventOfHits, mkFitGeom);

  mkfit::StdSeq::Cmssw_LoadHits_End(*eventOfHits);

  iEvent.emplace(putToken_, std::move(eventOfHits));
}

void MkFitEventOfHitsProducer::fill(const std::vector<const TrackingRecHit*>& hits,
                                    mkfit::EventOfHits& eventOfHits,
                                    const MkFitGeometry& mkFitGeom) const {
  for (int i = 0, end = hits.size(); i < end; ++i) {
    const auto* hit = hits[i];
    if (hit != nullptr) {
      const auto ilay = mkFitGeom.mkFitLayerNumber(hit->geographicalId());
      eventOfHits[ilay].RegisterHit(i);
    }
  }
}

DEFINE_FWK_MODULE(MkFitEventOfHitsProducer);
