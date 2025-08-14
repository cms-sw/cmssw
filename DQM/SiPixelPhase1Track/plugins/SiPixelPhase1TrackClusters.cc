// -*- C++ -*-
//
// Package:     SiPixelPhase1TrackClusters
// Class  :     SiPixelPhase1TrackClusters
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"
#include "CalibTracker/Records/interface/SiPixelTemplateDBObjectESProducerRcd.h"

namespace {

  class SiPixelPhase1TrackClusters final : public SiPixelPhase1Base {
    enum {
      ON_TRACK_CHARGE,
      ON_TRACK_CORRECTEDCHARGE,
      TEMPLATE_CORRECTION,
      ON_TRACK_BIGPIXELCHARGE,
      ON_TRACK_NOTBIGPIXELCHARGE,
      ON_TRACK_SIZE,
      ON_TRACK_SHAPE,
      ON_TRACK_NCLUSTERS,
      ON_TRACK_POSITIONB,
      ON_TRACK_POSITIONF,
      DIGIS_HITMAP_ON_TRACK,
      ON_TRACK_NDIGIS,

      NTRACKS,
      NTRACKS_INVOLUME,

      SIZE_VS_ETA_ON_TRACK_OUTER,
      SIZE_VS_ETA_ON_TRACK_INNER,
      ON_TRACK_CHARGE_OUTER,
      ON_TRACK_CHARGE_INNER,
      ON_TRACK_CORRECTEDCHARGE_OUTER,
      ON_TRACK_CORRECTEDCHARGE_INNER,
      TEMPLATE_CORRECTION_OUTER,
      TEMPLATE_CORRECTION_INNER,

      ON_TRACK_SHAPE_OUTER,
      ON_TRACK_SHAPE_INNER,

      ON_TRACK_SIZE_X_OUTER,
      ON_TRACK_SIZE_X_INNER,
      ON_TRACK_SIZE_X_F,
      ON_TRACK_SIZE_Y_OUTER,
      ON_TRACK_SIZE_Y_INNER,
      ON_TRACK_SIZE_Y_F,

      ON_TRACK_SIZE_XY_OUTER,
      ON_TRACK_SIZE_XY_INNER,
      ON_TRACK_SIZE_XY_F,
      CHARGE_VS_SIZE_ON_TRACK,

      ENUM_SIZE
    };

  public:
    explicit SiPixelPhase1TrackClusters(const edm::ParameterSet& conf);
    void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    const bool applyVertexCut_;
    const SiPixelTemplateDBObject* templateDBobject_ = nullptr;
    const std::vector<SiPixelTemplateStore>* thePixelTemp_ = nullptr;
    const TrackerTopology* tkTpl = nullptr;

    edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
    edm::EDGetTokenT<reco::VertexCollection> offlinePrimaryVerticesToken_;
    edm::EDGetTokenT<SiPixelClusterShapeCache> pixelClusterShapeCacheToken_;

    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
    edm::ESGetToken<ClusterShapeHitFilter, CkfComponentsRecord> clusterShapeHitFilterToken_;

    edm::ESGetToken<SiPixelTemplateDBObject, SiPixelTemplateDBObjectESProducerRcd> templateDBobjectToken_;
    edm::ESGetToken<std::vector<SiPixelTemplateStore>, SiPixelTemplateDBObjectESProducerRcd> templateStoreToken_;
  };

  SiPixelPhase1TrackClusters::SiPixelPhase1TrackClusters(const edm::ParameterSet& iConfig)
      : SiPixelPhase1Base(iConfig), applyVertexCut_(iConfig.getUntrackedParameter<bool>("VertexCut", true)) {
    tracksToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));

    offlinePrimaryVerticesToken_ =
        applyVertexCut_ ? consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))
                        : edm::EDGetTokenT<reco::VertexCollection>();

    pixelClusterShapeCacheToken_ =
        consumes<SiPixelClusterShapeCache>(iConfig.getParameter<edm::InputTag>("clusterShapeCache"));

    trackerTopoToken_ = esConsumes<edm::Transition::BeginRun>();
    trackerGeomToken_ = esConsumes<edm::Transition::BeginRun>();
    clusterShapeHitFilterToken_ =
        esConsumes<ClusterShapeHitFilter, CkfComponentsRecord>(edm::ESInputTag("", "ClusterShapeHitFilter"));
    templateStoreToken_ = esConsumes<edm::Transition::BeginRun>();
    templateDBobjectToken_ = esConsumes<edm::Transition::BeginRun>();
  }

  void SiPixelPhase1TrackClusters::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
    //get topology
    tkTpl = &iSetup.getData(trackerTopoToken_);

    // Initialize 1D templates
    templateDBobject_ = &iSetup.getData(templateDBobjectToken_);
    thePixelTemp_ = &iSetup.getData(templateStoreToken_);
  }

  void SiPixelPhase1TrackClusters::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    if (!checktrigger(iEvent, iSetup, DCS))
      return;

    if (histo.size() != ENUM_SIZE) {
      edm::LogError("SiPixelPhase1TrackClusters")
          << "incompatible configuration " << histo.size() << "!=" << ENUM_SIZE << std::endl;
      return;
    }

    edm::ESHandle<ClusterShapeHitFilter> shapeFilterH = iSetup.getHandle(clusterShapeHitFilterToken_);
    auto const& shapeFilter = *shapeFilterH;

    edm::Handle<reco::VertexCollection> vertices;
    if (applyVertexCut_) {
      iEvent.getByToken(offlinePrimaryVerticesToken_, vertices);
      if (!vertices.isValid() || vertices->empty())
        return;
    }

    //get the map
    edm::Handle<reco::TrackCollection> tracks;
    iEvent.getByToken(tracksToken_, tracks);

    if (!tracks.isValid()) {
      edm::LogWarning("SiPixelPhase1TrackClusters") << "track collection is not valid";
      return;
    }

    SiPixelTemplate templ(*thePixelTemp_);

    edm::Handle<SiPixelClusterShapeCache> pixelClusterShapeCacheH;
    iEvent.getByToken(pixelClusterShapeCacheToken_, pixelClusterShapeCacheH);
    if (!pixelClusterShapeCacheH.isValid()) {
      edm::LogWarning("SiPixelPhase1TrackClusters") << "PixelClusterShapeCache collection is not valid";
      return;
    }
    auto const& pixelClusterShapeCache = *pixelClusterShapeCacheH;

    for (auto const& track : *tracks) {
      if (applyVertexCut_ &&
          (track.pt() < 0.75 || std::abs(track.dxy((*vertices)[0].position())) > 5 * track.dxyError()))
        continue;

      bool isBpixtrack = false, isFpixtrack = false, crossesPixVol = false;

      // find out whether track crosses pixel fiducial volume (for cosmic tracks)
      auto d0 = track.d0(), dz = track.dz();
      if (std::abs(d0) < 16 && std::abs(dz) < 50)
        crossesPixVol = true;

      auto etatk = track.eta();

      auto const& trajParams = track.extra()->trajParams();
      assert(trajParams.size() == track.recHitsSize());
      auto hb = track.recHitsBegin();

      for (unsigned int h = 0; h < track.recHitsSize(); h++) {
        auto hit = *(hb + h);
        if (!hit->isValid())
          continue;
        auto id = hit->geographicalId();

        // check that we are in the pixel
        auto subdetid = (id.subdetId());
        if (subdetid == PixelSubdetector::PixelBarrel)
          isBpixtrack = true;
        if (subdetid == PixelSubdetector::PixelEndcap)
          isFpixtrack = true;
        if (subdetid != PixelSubdetector::PixelBarrel && subdetid != PixelSubdetector::PixelEndcap)
          continue;
        bool iAmBarrel = subdetid == PixelSubdetector::PixelBarrel;

        // PXB_L4 IS IN THE OTHER WAY
        // CAN BE XORed BUT LETS KEEP THINGS SIMPLE
        bool iAmOuter = ((tkTpl->pxbLadder(id) % 2 == 1) && tkTpl->pxbLayer(id) != 4) ||
                        ((tkTpl->pxbLadder(id) % 2 != 1) && tkTpl->pxbLayer(id) == 4);

        auto pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
        if (!pixhit)
          continue;

        auto geomdetunit = dynamic_cast<const PixelGeomDetUnit*>(pixhit->detUnit());
        auto const& topol = geomdetunit->specificTopology();

        // get the cluster
        auto clustp = pixhit->cluster();
        if (clustp.isNull())
          continue;
        auto const& cluster = *clustp;
        const std::vector<SiPixelCluster::Pixel> pixelsVec = cluster.pixels();
        for (unsigned int i = 0; i < pixelsVec.size(); ++i) {
          float pixx = pixelsVec[i].x;  // index as float=iteger, row index
          float pixy = pixelsVec[i].y;  // same, col index

          bool bigInX = topol.isItBigPixelInX(int(pixx));
          bool bigInY = topol.isItBigPixelInY(int(pixy));
          float pixel_charge = pixelsVec[i].adc;

          if (bigInX == true || bigInY == true) {
            histo[ON_TRACK_BIGPIXELCHARGE].fill(pixel_charge, id, &iEvent);
          } else {
            histo[ON_TRACK_NOTBIGPIXELCHARGE].fill(pixel_charge, id, &iEvent);
          }
        }  // End loop over pixels
        auto const& ltp = trajParams[h];

        auto localDir = ltp.momentum() / ltp.momentum().mag();

        // correct charge for track impact angle
        auto charge = cluster.charge() * ltp.absdz();
        //Correct charge with Template1D
        float cotAlpha = ltp.dxdz();
        float cotBeta = ltp.dydz();
        float locBx = 1.;
        if (cotBeta < 0.)
          locBx = -1.;
        float locBz = locBx;
        if (cotAlpha < 0.)
          locBz = -locBx;
        templ.interpolate(templateDBobject_->getTemplateID(id), cotAlpha, cotBeta, locBz, locBx);
        auto charge_cor = (charge * templ.qscale()) / templ.r_qMeas_qTrue();
        auto tmpl = templ.qscale() / templ.r_qMeas_qTrue();

        auto clustgp = pixhit->globalPosition();  // from rechit

        int part;
        ClusterData::ArrayType meas;
        std::pair<float, float> pred;
        if (shapeFilter.getSizes(*pixhit, localDir, pixelClusterShapeCache, part, meas, pred)) {
          auto shape = shapeFilter.isCompatible(*pixhit, localDir, pixelClusterShapeCache);
          unsigned shapeVal = (shape ? 1 : 0);

          if (iAmBarrel) {
            if (iAmOuter) {
              histo[ON_TRACK_SIZE_X_OUTER].fill(pred.first, cluster.sizeX(), id, &iEvent);
              histo[ON_TRACK_SIZE_Y_OUTER].fill(pred.second, cluster.sizeY(), id, &iEvent);
              histo[ON_TRACK_SIZE_XY_OUTER].fill(cluster.sizeY(), cluster.sizeX(), id, &iEvent);

              histo[ON_TRACK_SHAPE_OUTER].fill(shapeVal, id, &iEvent);
            } else {
              histo[ON_TRACK_SIZE_X_INNER].fill(pred.first, cluster.sizeX(), id, &iEvent);
              histo[ON_TRACK_SIZE_Y_INNER].fill(pred.second, cluster.sizeY(), id, &iEvent);
              histo[ON_TRACK_SIZE_XY_INNER].fill(cluster.sizeY(), cluster.sizeX(), id, &iEvent);

              histo[ON_TRACK_SHAPE_INNER].fill(shapeVal, id, &iEvent);
            }
          } else {
            histo[ON_TRACK_SIZE_X_F].fill(pred.first, cluster.sizeX(), id, &iEvent);
            histo[ON_TRACK_SIZE_Y_F].fill(pred.second, cluster.sizeY(), id, &iEvent);
            histo[ON_TRACK_SIZE_XY_F].fill(cluster.sizeY(), cluster.sizeX(), id, &iEvent);
          }
          histo[ON_TRACK_SHAPE].fill(shapeVal, id, &iEvent);
        }

        for (int i = 0; i < cluster.size(); i++) {
          SiPixelCluster::Pixel const& vecipxl = cluster.pixel(i);
          histo[DIGIS_HITMAP_ON_TRACK].fill(id, &iEvent, vecipxl.y, vecipxl.x);
          histo[ON_TRACK_NDIGIS].fill(id, &iEvent);
        }

        histo[ON_TRACK_NCLUSTERS].fill(id, &iEvent);
        histo[ON_TRACK_CHARGE].fill(charge, id, &iEvent);
        histo[ON_TRACK_CORRECTEDCHARGE].fill(charge_cor, id, &iEvent);
        histo[TEMPLATE_CORRECTION].fill(tmpl, id, &iEvent);
        histo[ON_TRACK_SIZE].fill(cluster.size(), id, &iEvent);

        histo[ON_TRACK_POSITIONB].fill(clustgp.z(), clustgp.phi(), id, &iEvent);
        histo[ON_TRACK_POSITIONF].fill(clustgp.x(), clustgp.y(), id, &iEvent);

        histo[CHARGE_VS_SIZE_ON_TRACK].fill(cluster.size(), charge, id, &iEvent);

        if (iAmBarrel)  // Avoid mistakes even if specification < should > handle it
        {
          if (iAmOuter) {
            histo[SIZE_VS_ETA_ON_TRACK_OUTER].fill(etatk, cluster.sizeY(), id, &iEvent);
            histo[ON_TRACK_CHARGE_OUTER].fill(charge, id, &iEvent);
            histo[ON_TRACK_CORRECTEDCHARGE_OUTER].fill(charge_cor, id, &iEvent);
            histo[TEMPLATE_CORRECTION_OUTER].fill(tmpl, id, &iEvent);
          } else {
            histo[SIZE_VS_ETA_ON_TRACK_INNER].fill(etatk, cluster.sizeY(), id, &iEvent);
            histo[ON_TRACK_CHARGE_INNER].fill(charge, id, &iEvent);
            histo[ON_TRACK_CORRECTEDCHARGE_INNER].fill(charge_cor, id, &iEvent);
            histo[TEMPLATE_CORRECTION_INNER].fill(tmpl, id, &iEvent);
          }
        }
      }

      // statistics on tracks
      histo[NTRACKS].fill(1, DetId(0), &iEvent);
      if (isBpixtrack || isFpixtrack)
        histo[NTRACKS].fill(2, DetId(0), &iEvent);
      if (isBpixtrack)
        histo[NTRACKS].fill(3, DetId(0), &iEvent);
      if (isFpixtrack)
        histo[NTRACKS].fill(4, DetId(0), &iEvent);

      if (crossesPixVol) {
        if (isBpixtrack || isFpixtrack)
          histo[NTRACKS_INVOLUME].fill(1, DetId(0), &iEvent);
        else
          histo[NTRACKS_INVOLUME].fill(0, DetId(0), &iEvent);
      }
    }

    histo[ON_TRACK_NCLUSTERS].executePerEventHarvesting(&iEvent);
    histo[ON_TRACK_NDIGIS].executePerEventHarvesting(&iEvent);
  }

}  // namespace

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPixelPhase1TrackClusters);
