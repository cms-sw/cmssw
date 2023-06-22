#include "CalibFormats/SiStripObjects/interface/SiStripDetInfo.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"
#include "RecoTracker/PixelLowPtUtilities/interface/SlidingPeakFinder.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include <vector>
#include <memory>

class SiStripClusters2ApproxClusters : public edm::stream::EDProducer<> {
public:
  explicit SiStripClusters2ApproxClusters(const edm::ParameterSet& conf);
  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::InputTag inputClusters;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusterToken;

  unsigned int maxNSat;
  static constexpr double subclusterWindow_ = .7;
  static constexpr double seedCutMIPs_ = .35;
  static constexpr double seedCutSN_ = 7.;
  static constexpr double subclusterCutMIPs_ = .45;
  static constexpr double subclusterCutSN_ = 12.;

  edm::InputTag beamSpot_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;

  edm::FileInPath fileInPath_;
  SiStripDetInfo detInfo_;

  std::string csfLabel_;
  edm::ESGetToken<ClusterShapeHitFilter, CkfComponentsRecord> csfToken_;

  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> stripNoiseToken_;
  edm::ESHandle<SiStripNoises> theNoise_;
};

SiStripClusters2ApproxClusters::SiStripClusters2ApproxClusters(const edm::ParameterSet& conf) {
  inputClusters = conf.getParameter<edm::InputTag>("inputClusters");
  maxNSat = conf.getParameter<unsigned int>("maxSaturatedStrips");

  clusterToken = consumes<edmNew::DetSetVector<SiStripCluster> >(inputClusters);

  beamSpot_ = conf.getParameter<edm::InputTag>("beamSpot");
  beamSpotToken_ = consumes<reco::BeamSpot>(beamSpot_);

  tkGeomToken_ = esConsumes();

  fileInPath_ = edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile);
  detInfo_ = SiStripDetInfoFileReader::read(fileInPath_.fullPath());

  csfLabel_ = conf.getParameter<std::string>("clusterShapeHitFilterLabel");
  csfToken_ = esConsumes(edm::ESInputTag("", csfLabel_));

  stripNoiseToken_ = esConsumes();

  produces<edmNew::DetSetVector<SiStripApproximateCluster> >();
}

void SiStripClusters2ApproxClusters::produce(edm::Event& event, edm::EventSetup const& iSetup) {
  auto result = std::make_unique<edmNew::DetSetVector<SiStripApproximateCluster> >();
  const auto& clusterCollection = event.get(clusterToken);

  //  edm::Handle<reco::BeamSpot> beamSpotHandle;
  //  event.getByToken(beamSpotToken, beamSpotHandle);  // retrive BeamSpot data
  //  reco::BeamSpot const* bs = nullptr;
  //  if (beamSpotHandle.isValid()) {
  //    bs = &(*beamSpotHandle);
  //  } else {
  //    edm::LogError("SiStripClusters2ApproxClusters")
  //        << "didn't find a valid beamspot with label " << beamSpot.label() << " using 0,0,0";
  //    bs = new reco::BeamSpot();
  //  }

  auto const beamSpotHandle = event.getHandle(beamSpotToken_);
  auto const& bs = beamSpotHandle.isValid() ? *beamSpotHandle : reco::BeamSpot();
  if (not beamSpotHandle.isValid()) {
    edm::LogError("SiStripClusters2ApproxClusters")
        << "didn't find a valid beamspot with label \"" << beamSpot_.encode() << "\" -> using (0,0,0)";
  }

  const auto& tkGeom = &iSetup.getData(tkGeomToken_);
  const auto& theFilter = &iSetup.getData(csfToken_);
  const auto& theNoise_ = &iSetup.getData(stripNoiseToken_);

  for (const auto& detClusters : clusterCollection) {
    edmNew::DetSetVector<SiStripApproximateCluster>::FastFiller ff{*result, detClusters.id()};

    unsigned int detId = detClusters.id();
    const GeomDet* det = tkGeom->idToDet(detId);
    double nApvs = detInfo_.getNumberOfApvsAndStripLength(detId).first;
    double stripLength = detInfo_.getNumberOfApvsAndStripLength(detId).second;
    double barycenter_ypos = 0.5 * stripLength;

    const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(det);
    float mip = 3.9 / (sistrip::MeVperADCStrip / stripDet->surface().bounds().thickness());

    for (const auto& cluster : detClusters) {
      const LocalPoint& lp = LocalPoint(((cluster.barycenter() * 10 / (sistrip::STRIPS_PER_APV * nApvs)) -
                                         ((stripDet->surface().bounds().width()) * 0.5f)),
                                        barycenter_ypos - (0.5f * stripLength),
                                        0.);
      const GlobalPoint& gpos = det->surface().toGlobal(lp);
      GlobalPoint beamspot(bs.position().x(), bs.position().y(), bs.position().z());
      const GlobalVector& gdir = gpos - beamspot;
      const LocalVector& ldir = det->toLocal(gdir);

      int hitStrips;
      float hitPredPos;
      theFilter->getSizes(detId, cluster, lp, ldir, hitStrips, hitPredPos);

      bool peakFilter = false;
      SlidingPeakFinder pf(std::max<int>(2, std::ceil(std::abs(hitPredPos) + subclusterWindow_)));
      float mipnorm = mip / std::abs(ldir.z());
      PeakFinderTest test(mipnorm,
                          detId,
                          cluster.firstStrip(),
                          theNoise_,
                          seedCutMIPs_,
                          seedCutSN_,
                          subclusterCutMIPs_,
                          subclusterCutSN_);
      peakFilter = pf.apply(cluster.amplitudes(), test);

      ff.push_back(SiStripApproximateCluster(cluster, maxNSat, hitPredPos, peakFilter));
    }
  }

  event.put(std::move(result));
}

void SiStripClusters2ApproxClusters::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputClusters", edm::InputTag("siStripClusters"));
  desc.add<unsigned int>("maxSaturatedStrips", 3);
  desc.add<std::string>("clusterShapeHitFilterLabel", "ClusterShapeHitFilter");  // add CSF label
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));         // add BeamSpot tag
  descriptions.add("SiStripClusters2ApproxClusters", desc);
}

DEFINE_FWK_MODULE(SiStripClusters2ApproxClusters);
