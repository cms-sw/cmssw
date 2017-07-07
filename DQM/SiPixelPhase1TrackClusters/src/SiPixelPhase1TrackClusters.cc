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


namespace {

class SiPixelPhase1TrackClusters final : public SiPixelPhase1Base {
  enum {
    ONTRACK_CHARGE,
    ONTRACK_SIZE,
    ONTRACK_NCLUSTERS,
    ONTRACK_POSITION_B,
    ONTRACK_POSITION_F,

    NTRACKS,
    NTRACKS_VOLUME,
    ONTRACK_SIZE_VS_ETA

  };

public:
  explicit SiPixelPhase1TrackClusters(const edm::ParameterSet& conf);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const bool applyVertexCut_;

  const edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > clustersToken_;
  const edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  const edm::EDGetTokenT<reco::VertexCollection> offlinePrimaryVerticesToken_;
};



SiPixelPhase1TrackClusters::SiPixelPhase1TrackClusters(const edm::ParameterSet& iConfig) :
  SiPixelPhase1Base(iConfig),
  applyVertexCut_(iConfig.getUntrackedParameter<bool>("VertexCut",true)),
  tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))),
  offlinePrimaryVerticesToken_(applyVertexCut_ ?
                              consumes<reco::VertexCollection>(std::string("offlinePrimaryVertices")) :
                              edm::EDGetTokenT<reco::VertexCollection>())
{}

void SiPixelPhase1TrackClusters::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  if (histo.size() != ONTRACK_SIZE_VS_ETA+1) {
    edm::LogError("SiPixelPhase1TrackClusters") << "incompatible configuration " << histo.size()
         << '<' << ONTRACK_SIZE_VS_ETA+1 << std::endl;
    return;
  }

  // get geometry
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  assert(tracker.isValid());

   edm::ESHandle<TrackerTopology> tTopoHandle;
   iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
   // auto const & tkTpl = *tTopoHandle;

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
      auto pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
      if (!pixhit) continue;

      // auto geomdetunit = dynamic_cast<const PixelGeomDetUnit*> (pixhit->detUnit());
      // auto const & topol = geomdetunit->specificTopology();
      
      // get the cluster
      auto clustp = pixhit->cluster();
      if (clustp.isNull()) continue; 
      auto const & cluster = *clustp;

      auto const & ltp = trajParams[h];
      
      // LocalVector localDir = ltp.momentum()/ltp.momentum().mag();

      // correct charge for track impact angle
      auto charge = cluster.charge()*ltp.absdz();

      auto clustgp =  pixhit->globalPosition();  // from rechit

      histo[ONTRACK_NCLUSTERS ].fill(id, &iEvent);
      histo[ONTRACK_CHARGE    ].fill(charge, id, &iEvent);
      histo[ONTRACK_SIZE	].fill(cluster.size(), id, &iEvent);
      histo[ONTRACK_POSITION_B].fill(clustgp.z(),   clustgp.phi(),   id, &iEvent);
      histo[ONTRACK_POSITION_F].fill(clustgp.x(),   clustgp.y(),     id, &iEvent);
      histo[ONTRACK_SIZE_VS_ETA].fill(etatk, cluster.sizeY(), id, &iEvent);
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
        histo[NTRACKS_VOLUME].fill(1, DetId(0), &iEvent);
      else 
        histo[NTRACKS_VOLUME].fill(0, DetId(0), &iEvent);
    }
  }

  histo[ONTRACK_NCLUSTERS].executePerEventHarvesting(&iEvent);
}

}// namespace

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPixelPhase1TrackClusters);

