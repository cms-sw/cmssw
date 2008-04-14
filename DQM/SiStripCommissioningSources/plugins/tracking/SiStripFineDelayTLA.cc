#include <memory>
#include <string>
#include <iostream>
#include <TMath.h>
#include "DQM/SiStripCommissioningSources/plugins/tracking/SiStripFineDelayTLA.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DQM/SiStripCommissioningSources/plugins/tracking/SimpleTrackRefitter.h"

using namespace std;
SiStripFineDelayTLA::SiStripFineDelayTLA(edm::ParameterSet const& conf) : 
  conf_(conf)
{
  refitter_ = new SimpleTrackRefitter(conf);
}

void SiStripFineDelayTLA::init(const edm::Event& e, const edm::EventSetup& es)
{
  // get geometry
  edm::ESHandle<TrackerGeometry> estracker;
  es.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(*estracker);
  // the refitter
  refitter_->setServices(es);
}

// Virtual destructor needed.
SiStripFineDelayTLA::~SiStripFineDelayTLA() 
{  
  delete refitter_;
}  

std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> > SiStripFineDelayTLA::findtrackangle(const reco::Track& theT)
{
  std::vector<Trajectory> traj = refitter_->refitTrack(theT);
  return findtrackangle(traj);
}

std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> > SiStripFineDelayTLA::findtrackangle(const TrajectorySeed& seed, const reco::Track& theT){
  vector<Trajectory> traj = refitter_->refitTrack(seed,theT);
  return findtrackangle(traj);
}       

std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> > SiStripFineDelayTLA::findtrackangle(const std::vector<Trajectory>& trajVec)
{
  if (trajVec.size()) {
  return findtrackangle(trajVec.front()); }
  std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> > hitangleassociation;
  return hitangleassociation;
}

std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> > SiStripFineDelayTLA::findtrackangle(const Trajectory& traj)
{
  std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> >hitangleassociation;
  std::vector<TrajectoryMeasurement> TMeas=traj.measurements();
  std::vector<TrajectoryMeasurement>::iterator itm;
  int i=0;
  LogDebug("SiStripFineDelayTLA::findtrackangle")<<"Loop on rechit and TSOS";
  for (itm=TMeas.begin();itm!=TMeas.end();itm++){
    //std::cout<<"hit: "<<i++<<std::endl;
    TrajectoryStateOnSurface tsos=itm->updatedState();
    const TransientTrackingRecHit::ConstRecHitPointer thit=itm->recHit();
    const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>((*thit).hit());
    const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>((*thit).hit());
    LocalVector trackdirection=tsos.localDirection();
    if(matchedhit){//if matched hit...
	GluedGeomDet * gdet=(GluedGeomDet *)tracker->idToDet(matchedhit->geographicalId());
	GlobalVector gtrkdir=gdet->toGlobal(trackdirection);
	//cluster and trackdirection on mono det
	// this the pointer to the mono hit of a matched hit 
	const SiStripRecHit2D *monohit=matchedhit->monoHit();
	const SiStripRecHit2D::ClusterRef & monocluster=monohit->cluster();
	const GeomDetUnit * monodet=gdet->monoDet();
	LocalVector monotkdir=monodet->toLocal(gtrkdir);
	if(monotkdir.z()!=0){
	  // the local angle (mono)
	  float angle = acos(monotkdir.z()/monotkdir.mag())*180/TMath::Pi();
	  hitangleassociation.push_back(make_pair(make_pair(monohit->geographicalId(),monohit->localPosition()), angle)); 
	  //cluster and trackdirection on stereo det
	  // this the pointer to the stereo hit of a matched hit 
	  const SiStripRecHit2D *stereohit=matchedhit->stereoHit();
	  const SiStripRecHit2D::ClusterRef & stereocluster=stereohit->cluster();
	  const GeomDetUnit * stereodet=gdet->stereoDet(); 
	  LocalVector stereotkdir=stereodet->toLocal(gtrkdir);
	  if(stereotkdir.z()!=0){
	    // the local angle (stereo)
	    float angle = acos(stereotkdir.z()/stereotkdir.mag())*180/TMath::Pi();
 	    hitangleassociation.push_back(make_pair(make_pair(stereohit->geographicalId(),stereohit->localPosition()), angle)); 
	  }
	}
    }
    else if(hit){
	//  hit= pointer to the rechit
	const SiStripRecHit2D::ClusterRef & cluster=hit->cluster();
	if(trackdirection.z()!=0){
	  float angle = acos(trackdirection.z()/trackdirection.mag())*180/TMath::Pi();
	  hitangleassociation.push_back(make_pair(make_pair(hit->geographicalId(),hit->localPosition()), angle)); 
	}
    }
    LogDebug("SiStripFineDelayTLA")<<"I found "<<i<<" hits.";
  }
  return hitangleassociation;
}

