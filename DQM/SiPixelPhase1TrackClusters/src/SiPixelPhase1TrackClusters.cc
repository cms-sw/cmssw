// -*- C++ -*-
//
// Package:     SiPixelPhase1TrackClusters
// Class:       SiPixelPhase1TrackClusters
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1TrackClusters/interface/SiPixelPhase1TrackClusters.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"


SiPixelPhase1TrackClusters::SiPixelPhase1TrackClusters(const edm::ParameterSet& iConfig) :
  SiPixelPhase1Base(iConfig) 
{
  clustersToken_ = consumes<edmNew::DetSetVector<SiPixelCluster>>(iConfig.getParameter<edm::InputTag>("clusters"));
  tracksToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("trajectories"));
  trajectoryToken_ = consumes<std::vector<Trajectory>>(iConfig.getParameter<edm::InputTag>("trajectories"));
  trackAssociationToken_ = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("trajectories"));
}

void SiPixelPhase1TrackClusters::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  assert(tracker.isValid());
  
  //get trajectories
  edm::Handle<std::vector<Trajectory> > trajCollectionHandle;
  iEvent.getByToken ( trajectoryToken_, trajCollectionHandle );
  auto const & trajColl = *(trajCollectionHandle.product());
   
  //get tracks
  edm::Handle<std::vector<reco::Track> > trackCollectionHandle;
  iEvent.getByToken( tracksToken_, trackCollectionHandle );
  auto const & trackColl = *(trackCollectionHandle.product());
  
  //get the map
  edm::Handle<TrajTrackAssociationCollection> match;
  iEvent.getByToken( trackAssociationToken_, match);
  auto const &  ttac = *(match.product());
  
  // get clusters
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> >  clusterColl;
  iEvent.getByToken( clustersToken_, clusterColl );
  auto const & clustColl = *(clusterColl.product());
  
  TrajectoryStateCombiner tsoscomb;

  for (auto& item : ttac) {
    auto trajectory_ref = item.key;
    reco::TrackRef track_ref = item.val;

    bool isBpixtrack = false, isFpixtrack = false, crossesPixVol=false;

    // find out whether track crosses pixel fiducial volume (for cosmic tracks)
    double d0 = track_ref->d0(), dz = track_ref->dz(); 
    if(std::abs(d0)<15 && std::abs(dz)<50) crossesPixVol = true;

    for (auto& measurement : trajectory_ref->measurements()) {
      // check if things are all valid
      if (!measurement.updatedState().isValid()) continue;
      auto hit = measurement.recHit();
      if (!hit->isValid()) continue;
      DetId id = hit->geographicalId();

      // check that we are in the pixel
      uint32_t subdetid = (id.subdetId());
      if (subdetid == PixelSubdetector::PixelBarrel) isBpixtrack = true;
      if (subdetid == PixelSubdetector::PixelEndcap) isFpixtrack = true;
      if (subdetid != PixelSubdetector::PixelBarrel && subdetid != PixelSubdetector::PixelEndcap) continue;
      auto pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
      if (!pixhit) continue;

      // get the cluster
      auto clust = pixhit->cluster();
      if (clust.isNull()) continue; 

      // get pixel topo (for local/global pos)
      const PixelGeomDetUnit* geomdetunit = static_cast<const PixelGeomDetUnit*> (tracker->idToDet(id));
      if (!geomdetunit) continue; 
      const PixelTopology& topol = geomdetunit->specificTopology();
      LocalPoint  clustlp = topol.localPosition (MeasurementPoint(clust->x(), clust->y()));
      GlobalPoint clustgp = geomdetunit->surface().toGlobal(clustlp);


      // compute trajectory parameters at hit
      TrajectoryStateOnSurface tsos = tsoscomb(measurement.forwardPredictedState(), 
                                               measurement.backwardPredictedState());
      if (!tsos.isValid()) continue;

      // correct charge for track impact angle
      LocalTrajectoryParameters ltp = tsos.localParameters();
      LocalVector localDir = ltp.momentum()/ltp.momentum().mag();
     
      float clust_alpha = atan2(localDir.z(), localDir.x());
      float clust_beta  = atan2(localDir.z(), localDir.y());
      double corrCharge = clust->charge() * sqrt( 1.0 / ( 1.0/pow( tan(clust_alpha), 2 ) + 
                                                          1.0/pow( tan(clust_beta ), 2 ) + 
                                                          1.0 ));

      // now we have all quantities at hand, fill on-track histograms now.
      // ...

    }
    // statistics on missing tracks
    if (crossesPixVol) {
      if (isBpixtrack || isFpixtrack) {
        //...
      } else {
        //...
      }
    }
  }

  edmNew::DetSetVector<SiPixelCluster>::const_iterator it;
  for (it = clustColl.begin(); it != clustColl.end(); ++it) {
    // TODO: check here if the cluster was seen above
    auto id = DetId(it->detId());

    const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*> ( tracker->idToDet(id) );
    const PixelTopology& topol = theGeomDet->specificTopology();

    for(SiPixelCluster const& cluster : *it) {
      histo[OFFTRACK_CHARGE].fill(double(cluster.charge()), id, &iEvent);
      histo[OFFTRACK_SIZE  ].fill(double(cluster.size()  ), id, &iEvent);
      histo[OFFTRACK_NCLUSTERS].fill(id, &iEvent);

      LocalPoint clustlp = topol.localPosition(MeasurementPoint(cluster.x(), cluster.y()));
      GlobalPoint clustgp = theGeomDet->surface().toGlobal(clustlp);
      histo[OFFTRACK_POSITION_B ].fill(clustgp.z(),   clustgp.phi(),   id, &iEvent);
      histo[OFFTRACK_POSITION_F ].fill(clustgp.x(),   clustgp.y(),     id, &iEvent);

    }
  }

  histo[ONTRACK_NCLUSTERS].executePerEventHarvesting();
  histo[OFFTRACK_NCLUSTERS].executePerEventHarvesting();
}

DEFINE_FWK_MODULE(SiPixelPhase1TrackClusters);

