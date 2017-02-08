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
  trackAssociationToken_ = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("trajectories"));
}

void SiPixelPhase1TrackClusters::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // get geometry
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  assert(tracker.isValid());
  
  //get the map
  edm::Handle<TrajTrackAssociationCollection> ttac;
  iEvent.getByToken( trackAssociationToken_, ttac);
  
  // get clusters
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> >  clusterColl;
  iEvent.getByToken( clustersToken_, clusterColl );
  
  TrajectoryStateCombiner tsoscomb;

  // we need to store some per-cluster data. Instead of a map, we use a vector,
  // exploiting the fact that all custers live in the DetSetVector and we can 
  // use the same indices to refer to them.
  // corr_charge is not strictly needed but cleaner to have it.
  std::vector<bool>  ontrack    (clusterColl->data().size(), false);
  std::vector<float> corr_charge(clusterColl->data().size(), -1.0f);

  for (auto& item : *ttac) {
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
      ontrack[clust.key()] = true; // mark cluster as ontrack

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
      corr_charge[clust.key()] = (float) corrCharge;
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

  edmNew::DetSetVector<SiPixelCluster>::const_iterator it;
  for (it = clusterColl->begin(); it != clusterColl->end(); ++it) {
    auto id = DetId(it->detId());

    const PixelGeomDetUnit* geomdetunit = dynamic_cast<const PixelGeomDetUnit*> ( tracker->idToDet(id) );
    const PixelTopology& topol = geomdetunit->specificTopology();

    for(auto subit = it->begin(); subit != it->end(); ++subit) {
      // we could do subit-...->data().front() as well, but this seems cleaner.
      auto key = edmNew::makeRefTo(clusterColl, subit).key(); 
      bool is_ontrack = ontrack[key];
      float corrected_charge = corr_charge[key];
      SiPixelCluster const& cluster = *subit;

      LocalPoint clustlp = topol.localPosition(MeasurementPoint(cluster.x(), cluster.y()));
      GlobalPoint clustgp = geomdetunit->surface().toGlobal(clustlp);

      if (is_ontrack) {
        histo[ONTRACK_NCLUSTERS ].fill(id, &iEvent);
        histo[ONTRACK_CHARGE    ].fill(double(corrected_charge), id, &iEvent);
        histo[ONTRACK_SIZE      ].fill(double(cluster.size()  ), id, &iEvent);
        histo[ONTRACK_POSITION_B].fill(clustgp.z(),   clustgp.phi(),   id, &iEvent);
        histo[ONTRACK_POSITION_F].fill(clustgp.x(),   clustgp.y(),     id, &iEvent);
      } else {
        histo[OFFTRACK_NCLUSTERS ].fill(id, &iEvent);
        histo[OFFTRACK_CHARGE    ].fill(double(cluster.charge()), id, &iEvent);
        histo[OFFTRACK_SIZE      ].fill(double(cluster.size()  ), id, &iEvent);
        histo[OFFTRACK_POSITION_B].fill(clustgp.z(),   clustgp.phi(),   id, &iEvent);
        histo[OFFTRACK_POSITION_F].fill(clustgp.x(),   clustgp.y(),     id, &iEvent);
      }
    }
  }

  histo[ONTRACK_NCLUSTERS].executePerEventHarvesting(&iEvent);
  histo[OFFTRACK_NCLUSTERS].executePerEventHarvesting(&iEvent);
}

DEFINE_FWK_MODULE(SiPixelPhase1TrackClusters);

