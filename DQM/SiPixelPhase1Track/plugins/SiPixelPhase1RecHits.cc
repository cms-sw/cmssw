// -*- C++ -*-
//
// Package:     SiPixelPhase1RecHits
// Class:       SiPixelPhase1RecHits
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace {

  class SiPixelPhase1RecHits final : public SiPixelPhase1Base {
    enum { NRECHITS, CLUST_X, CLUST_Y, ERROR_X, ERROR_Y, POS, CLUSTER_PROB, NONEDGE, NOTHERBAD };

  public:
    explicit SiPixelPhase1RecHits(const edm::ParameterSet& conf);
    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    edm::EDGetTokenT<reco::TrackCollection> srcToken_;
    edm::EDGetTokenT<reco::VertexCollection> offlinePrimaryVerticesToken_;

    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;

    bool onlyValid_;
    bool applyVertexCut_;
  };

  SiPixelPhase1RecHits::SiPixelPhase1RecHits(const edm::ParameterSet& iConfig) : SiPixelPhase1Base(iConfig) {
    srcToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("src"));

    offlinePrimaryVerticesToken_ = consumes<reco::VertexCollection>(std::string("offlinePrimaryVertices"));

    trackerGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();

    onlyValid_ = iConfig.getParameter<bool>("onlyValidHits");

    applyVertexCut_ = iConfig.getUntrackedParameter<bool>("VertexCut", true);
  }

  void SiPixelPhase1RecHits::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    if (!checktrigger(iEvent, iSetup, DCS))
      return;

    edm::ESHandle<TrackerGeometry> tracker = iSetup.getHandle(trackerGeomToken_);
    assert(tracker.isValid());

    edm::Handle<reco::TrackCollection> tracks;
    iEvent.getByToken(srcToken_, tracks);
    if (!tracks.isValid())
      return;

    edm::Handle<reco::VertexCollection> vertices;
    iEvent.getByToken(offlinePrimaryVerticesToken_, vertices);

    if (applyVertexCut_ && (!vertices.isValid() || vertices->empty()))
      return;

    for (auto const& track : *tracks) {
      if (applyVertexCut_ &&
          (track.pt() < 0.75 || std::abs(track.dxy(vertices->at(0).position())) > 5 * track.dxyError()))
        continue;

      bool isBpixtrack = false, isFpixtrack = false;

      auto const& trajParams = track.extra()->trajParams();
      auto hb = track.recHitsBegin();
      for (unsigned int h = 0; h < track.recHitsSize(); h++) {
        auto hit = *(hb + h);
        if (!trackerHitRTTI::isFromDet(*hit))
          continue;

        DetId id = hit->geographicalId();
        uint32_t subdetid = (id.subdetId());

        if (subdetid == PixelSubdetector::PixelBarrel)
          isBpixtrack = true;
        if (subdetid == PixelSubdetector::PixelEndcap)
          isFpixtrack = true;
      }

      if (!isBpixtrack && !isFpixtrack)
        continue;

      // then, look at each hit
      for (unsigned int h = 0; h < track.recHitsSize(); h++) {
        auto rechit = *(hb + h);

        if (!trackerHitRTTI::isFromDet(*rechit))
          continue;

        //continue if not a Pixel recHit
        DetId id = rechit->geographicalId();
        uint32_t subdetid = (id.subdetId());

        if (subdetid != PixelSubdetector::PixelBarrel && subdetid != PixelSubdetector::PixelEndcap)
          continue;

        bool isHitValid = rechit->getType() == TrackingRecHit::valid;
        if (onlyValid_ && !isHitValid)
          continue;  //useful to run on cosmics where the TrackEfficiency plugin is not used

        const SiPixelRecHit* prechit = dynamic_cast<const SiPixelRecHit*>(
            rechit);  //to be used to get the associated cluster and the cluster probability

        int sizeX = 0, sizeY = 0;

        if (isHitValid) {
          SiPixelRecHit::ClusterRef const& clust = prechit->cluster();
          sizeX = (*clust).sizeX();
          sizeY = (*clust).sizeY();
        }

        const PixelGeomDetUnit* geomdetunit = dynamic_cast<const PixelGeomDetUnit*>(tracker->idToDet(id));
        const PixelTopology& topol = geomdetunit->specificTopology();

        LocalPoint lp = trajParams[h].position();
        MeasurementPoint mp = topol.measurementPosition(lp);

        int row = (int)mp.x();
        int col = (int)mp.y();

        float rechit_x = lp.x();
        float rechit_y = lp.y();

        LocalError lerr = rechit->localPositionError();
        float lerr_x = sqrt(lerr.xx());
        float lerr_y = sqrt(lerr.yy());

        histo[NRECHITS].fill(id, &iEvent, col, row);  //in general a inclusive counter of missing/valid/inactive hits
        if (prechit->isOnEdge())
          histo[NONEDGE].fill(id, &iEvent, col, row);
        if (prechit->hasBadPixels())
          histo[NOTHERBAD].fill(id, &iEvent, col, row);

        if (isHitValid) {
          histo[CLUST_X].fill(sizeX, id, &iEvent, col, row);
          histo[CLUST_Y].fill(sizeY, id, &iEvent, col, row);
        }

        histo[ERROR_X].fill(lerr_x, id, &iEvent);
        histo[ERROR_Y].fill(lerr_y, id, &iEvent);

        histo[POS].fill(rechit_x, rechit_y, id, &iEvent);

        if (isHitValid) {
          double clusterProbability = prechit->clusterProbability(0);
          if (clusterProbability > 0)
            histo[CLUSTER_PROB].fill(log10(clusterProbability), id, &iEvent);
        }
      }
    }

    histo[NRECHITS].executePerEventHarvesting(&iEvent);
    histo[NONEDGE].executePerEventHarvesting(&iEvent);
    histo[NOTHERBAD].executePerEventHarvesting(&iEvent);
  }

}  //namespace

DEFINE_FWK_MODULE(SiPixelPhase1RecHits);
