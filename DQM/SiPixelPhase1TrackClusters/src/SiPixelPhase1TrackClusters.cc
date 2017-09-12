// -*- C++ -*-
// 
// Package:     SiPixelPhase1TrackClusters
// Class  :     SiPixelPhase1TrackClusters
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"

namespace {

class SiPixelPhase1TrackClusters final : public SiPixelPhase1Base {
enum {  // copy paste from cfy: the only safe way to doit....
  SiPixelPhase1TrackClustersOnTrackCharge,
  SiPixelPhase1TrackClustersOnTrackSize,
  SiPixelPhase1TrackClustersOnTrackShape,
  SiPixelPhase1TrackClustersOnTrackNClusters,
  SiPixelPhase1TrackClustersOnTrackPositionB,
  SiPixelPhase1TrackClustersOnTrackPositionF,

  SiPixelPhase1TrackClustersNTracks,
  SiPixelPhase1TrackClustersNTracksInVolume,

  SiPixelPhase1ClustersSizeVsEtaOnTrackOuter,
  SiPixelPhase1ClustersSizeVsEtaOnTrackInner,
  SiPixelPhase1TrackClustersOnTrackChargeOuter,
  SiPixelPhase1TrackClustersOnTrackChargeInner,

  SiPixelPhase1TrackClustersOnTrackShapeOuter,
  SiPixelPhase1TrackClustersOnTrackShapeInner,

  SiPixelPhase1TrackClustersOnTrackSizeXOuter,
  SiPixelPhase1TrackClustersOnTrackSizeXInner,
  SiPixelPhase1TrackClustersOnTrackSizeXF,
  SiPixelPhase1TrackClustersOnTrackSizeYOuter,
  SiPixelPhase1TrackClustersOnTrackSizeYInner,
  SiPixelPhase1TrackClustersOnTrackSizeYF,
    
  SiPixelPhase1TrackClustersOnTrackSizeXYOuter,
  SiPixelPhase1TrackClustersOnTrackSizeXYInner,
  SiPixelPhase1TrackClustersOnTrackSizeXYF,

  SiPixelPhase1TrackClustersEnumSize
};

public:
  explicit SiPixelPhase1TrackClusters(const edm::ParameterSet& conf);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const bool applyVertexCut_;

  const edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  const edm::EDGetTokenT<reco::VertexCollection> offlinePrimaryVerticesToken_;
  const edm::EDGetTokenT<SiPixelClusterShapeCache> pixelClusterShapeCacheToken_;
};



SiPixelPhase1TrackClusters::SiPixelPhase1TrackClusters(const edm::ParameterSet& iConfig) :
  SiPixelPhase1Base(iConfig),
  applyVertexCut_(iConfig.getUntrackedParameter<bool>("VertexCut",true)),
  tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))),
  offlinePrimaryVerticesToken_(applyVertexCut_ ?
                              consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices")) :
                              edm::EDGetTokenT<reco::VertexCollection>()),
  pixelClusterShapeCacheToken_(consumes<SiPixelClusterShapeCache>(iConfig.getParameter<edm::InputTag>("clusterShapeCache")))
{}

void SiPixelPhase1TrackClusters::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  if (histo.size() != SiPixelPhase1TrackClustersEnumSize) {
    edm::LogError("SiPixelPhase1TrackClusters") << "incompatible configuration " << histo.size()
         << "!=" << SiPixelPhase1TrackClustersEnumSize << std::endl;
    return;
  }

  // get geometry
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  assert(tracker.isValid());

   edm::ESHandle<TrackerTopology> tTopoHandle;
   iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
   auto const & tkTpl = *tTopoHandle;

   edm::ESHandle<ClusterShapeHitFilter> shapeFilterH;
   iSetup.get<CkfComponentsRecord>().get("ClusterShapeHitFilter", shapeFilterH);
   auto const & shapeFilter = *shapeFilterH;


  edm::Handle<reco::VertexCollection> vertices;
  if(applyVertexCut_) {
    iEvent.getByToken(offlinePrimaryVerticesToken_, vertices);
    if (!vertices.isValid() || vertices->empty()) return;
  }


  //get the map
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken( tracksToken_, tracks);

  if ( !tracks.isValid() ) {
    edm::LogWarning("SiPixelPhase1TrackClusters")  << "track collection is not valid";
    return;
  }

  edm::Handle<SiPixelClusterShapeCache> pixelClusterShapeCacheH;
  iEvent.getByToken(pixelClusterShapeCacheToken_, pixelClusterShapeCacheH);
  if ( !pixelClusterShapeCacheH.isValid() ) {
    edm::LogWarning("SiPixelPhase1TrackClusters")  << "PixelClusterShapeCache collection is not valid";
    return;
  }  
  auto const & pixelClusterShapeCache = *pixelClusterShapeCacheH;

  
  for (auto const & track : *tracks) {

    if (applyVertexCut_ && (track.pt() < 0.75 || std::abs( track.dxy((*vertices)[0].position()) ) > 5*track.dxyError())) continue;

    bool isBpixtrack = false, isFpixtrack = false, crossesPixVol=false;

    // find out whether track crosses pixel fiducial volume (for cosmic tracks)
    auto d0 = track.d0(), dz = track.dz(); 
    if(std::abs(d0)<15 && std::abs(dz)<50) crossesPixVol = true;

    auto etatk = track.eta();


    auto const & trajParams = track.extra()->trajParams();
    assert(trajParams.size()==track.recHitsSize());
    auto hb = track.recHitsBegin();
    for(unsigned int h=0;h<track.recHitsSize();h++){
      auto hit = *(hb+h);
      if (!hit->isValid()) continue;
      auto id = hit->geographicalId();

      // check that we are in the pixel
      auto subdetid = (id.subdetId());
      if (subdetid == PixelSubdetector::PixelBarrel) isBpixtrack = true;
      if (subdetid == PixelSubdetector::PixelEndcap) isFpixtrack = true;
      if (subdetid != PixelSubdetector::PixelBarrel && subdetid != PixelSubdetector::PixelEndcap) continue;
      bool iAmBarrel = subdetid ==PixelSubdetector::PixelBarrel;
      auto pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
      if (!pixhit) continue;

      // auto geomdetunit = dynamic_cast<const PixelGeomDetUnit*> (pixhit->detUnit());
      // auto const & topol = geomdetunit->specificTopology();
      
      // get the cluster
      auto clustp = pixhit->cluster();
      if (clustp.isNull()) continue; 
      auto const & cluster = *clustp;

      auto const & ltp = trajParams[h];
      
      auto localDir = ltp.momentum()/ltp.momentum().mag();

      // correct charge for track impact angle
      auto charge = cluster.charge()*ltp.absdz();

      auto clustgp =  pixhit->globalPosition();  // from rechit


      int part;
      ClusterData::ArrayType meas;
      std::pair<float,float> pred;
      if(shapeFilter.getSizes(*pixhit,localDir,pixelClusterShapeCache, part,meas, pred)) {
       auto shape = shapeFilter.isCompatible(*pixhit,localDir,pixelClusterShapeCache);
       if (iAmBarrel) {
         if(tkTpl.pxbLadder(id)%2==1) {
           histo[SiPixelPhase1TrackClustersOnTrackSizeXOuter].fill(pred.first, cluster.sizeX(), id, &iEvent);
           histo[SiPixelPhase1TrackClustersOnTrackSizeYOuter].fill(pred.second,cluster.sizeY(), id, &iEvent);
           histo[SiPixelPhase1TrackClustersOnTrackSizeXYOuter].fill(cluster.sizeY(),cluster.sizeX(), id, &iEvent);

           histo[SiPixelPhase1TrackClustersOnTrackShapeOuter].fill(shape?1:0,id, &iEvent);
         } else {
           histo[SiPixelPhase1TrackClustersOnTrackSizeXInner].fill(pred.first, cluster.sizeX(), id, &iEvent);
           histo[SiPixelPhase1TrackClustersOnTrackSizeYInner].fill(pred.second,cluster.sizeY(), id, &iEvent);
           histo[SiPixelPhase1TrackClustersOnTrackSizeXYInner].fill(cluster.sizeY(),cluster.sizeX(), id, &iEvent);

           histo[SiPixelPhase1TrackClustersOnTrackShapeInner].fill(shape?1:0,id, &iEvent);
         }
       } else {
           histo[SiPixelPhase1TrackClustersOnTrackSizeXF].fill(pred.first, cluster.sizeX(), id, &iEvent);
           histo[SiPixelPhase1TrackClustersOnTrackSizeYF].fill(pred.second,cluster.sizeY(), id, &iEvent);
           histo[SiPixelPhase1TrackClustersOnTrackSizeXYF].fill(cluster.sizeY(),cluster.sizeX(), id, &iEvent);
       }
       histo[SiPixelPhase1TrackClustersOnTrackShape].fill(shape?1:0,id, &iEvent);
      }

      histo[SiPixelPhase1TrackClustersOnTrackNClusters].fill(id, &iEvent);
      histo[SiPixelPhase1TrackClustersOnTrackCharge].fill(charge, id, &iEvent);
      histo[SiPixelPhase1TrackClustersOnTrackSize].fill(cluster.size(), id, &iEvent);

      histo[SiPixelPhase1TrackClustersOnTrackPositionB].fill(clustgp.z(),   clustgp.phi(),   id, &iEvent);
      histo[SiPixelPhase1TrackClustersOnTrackPositionF].fill(clustgp.x(),   clustgp.y(),     id, &iEvent);

      if(tkTpl.pxbLadder(id)%2==1) {
        histo[SiPixelPhase1ClustersSizeVsEtaOnTrackOuter].fill(etatk, cluster.sizeY(), id, &iEvent);
        histo[SiPixelPhase1TrackClustersOnTrackChargeOuter].fill(charge, id, &iEvent);
      } else {
        histo[SiPixelPhase1ClustersSizeVsEtaOnTrackInner].fill(etatk, cluster.sizeY(), id, &iEvent);
        histo[SiPixelPhase1TrackClustersOnTrackChargeInner].fill(charge, id, &iEvent);
      }


    }

    // statistics on tracks
    histo[SiPixelPhase1TrackClustersNTracks].fill(1, DetId(0), &iEvent);
    if (isBpixtrack || isFpixtrack) 
      histo[SiPixelPhase1TrackClustersNTracks].fill(2, DetId(0), &iEvent);
    if (isBpixtrack) 
      histo[SiPixelPhase1TrackClustersNTracks].fill(3, DetId(0), &iEvent);
    if (isFpixtrack) 
      histo[SiPixelPhase1TrackClustersNTracks].fill(4, DetId(0), &iEvent);

    if (crossesPixVol) {
      if (isBpixtrack || isFpixtrack)
        histo[SiPixelPhase1TrackClustersNTracksInVolume].fill(1, DetId(0), &iEvent);
      else 
        histo[SiPixelPhase1TrackClustersNTracksInVolume].fill(0, DetId(0), &iEvent);
    }
  }

  histo[SiPixelPhase1TrackClustersOnTrackNClusters].executePerEventHarvesting(&iEvent);
}

}// namespace

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPixelPhase1TrackClusters);

