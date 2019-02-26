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
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

using namespace std;
SiStripFineDelayTLA::SiStripFineDelayTLA(edm::ParameterSet const& conf) : 
  conf_(conf)
{
}

void SiStripFineDelayTLA::init(const edm::Event& e, const edm::EventSetup& es)
{
  // get geometry
  edm::ESHandle<TrackerGeometry> estracker;
  es.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(*estracker);
}

// Virtual destructor needed.
SiStripFineDelayTLA::~SiStripFineDelayTLA() 
{  
}  

std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> > SiStripFineDelayTLA::findtrackangle(const std::vector<Trajectory>& trajVec)
{
  if (!trajVec.empty()) {
  return findtrackangle(trajVec.front()); }
  std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> > hitangleassociation;
  return hitangleassociation;
}

std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> > SiStripFineDelayTLA::findtrackangle(const Trajectory& traj)
{
  std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> >hitangleassociation;
  std::vector<TrajectoryMeasurement> TMeas=traj.measurements();
  std::vector<TrajectoryMeasurement>::iterator itm;
  for (itm=TMeas.begin();itm!=TMeas.end();itm++){
    TrajectoryStateOnSurface tsos=itm->updatedState();
    auto thit=itm->recHit();
    const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>((*thit).hit());
    const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>((*thit).hit());
    LocalVector trackdirection=tsos.localDirection();
    if(matchedhit){//if matched hit...
	const GluedGeomDet * gdet=static_cast<const GluedGeomDet *>(tracker->idToDet(matchedhit->geographicalId()));
	GlobalVector gtrkdir=gdet->toGlobal(trackdirection);
	// trackdirection on mono det
	// this the pointer to the mono hit of a matched hit 
	const SiStripRecHit2D monohit=matchedhit->monoHit();
	const GeomDetUnit * monodet=gdet->monoDet();
	LocalVector monotkdir=monodet->toLocal(gtrkdir);
	if(monotkdir.z()!=0){
	  // the local angle (mono)
          float localpitch = static_cast<const StripTopology&>(monodet->topology()).localPitch(tsos.localPosition());
          float thickness = ((((((monohit.geographicalId())>>25)&0x7f)==0xd)||
	                     ((((monohit.geographicalId())>>25)&0x7f)==0xe))&&
			           ((((monohit.geographicalId())>>5)&0x7)>4)) ? 0.0500 : 0.0320;
          float angle = computeAngleCorr(monotkdir, localpitch, thickness);
	  hitangleassociation.push_back(make_pair(make_pair(monohit.geographicalId(),monohit.localPosition()), angle)); 
	  // trackdirection on stereo det
	  // this the pointer to the stereo hit of a matched hit 
	  const SiStripRecHit2D stereohit=matchedhit->stereoHit();
	  const GeomDetUnit * stereodet=gdet->stereoDet(); 
	  LocalVector stereotkdir=stereodet->toLocal(gtrkdir);
	  if(stereotkdir.z()!=0){
	    // the local angle (stereo)
            float localpitch = static_cast<const StripTopology&>(stereodet->topology()).localPitch(tsos.localPosition());
            float thickness = ((((((stereohit.geographicalId())>>25)&0x7f)==0xd)||
	                       ((((stereohit.geographicalId())>>25)&0x7f)==0xe))&&
			             ((((stereohit.geographicalId())>>5)&0x7)>4)) ? 0.0500 : 0.0320;
            float angle = computeAngleCorr(stereotkdir, localpitch, thickness);
 	    hitangleassociation.push_back(make_pair(make_pair(stereohit.geographicalId(),stereohit.localPosition()), angle)); 
	  }
	}
    }
    else if(hit){
	const GeomDetUnit * det=tracker->idToDet(hit->geographicalId());
	//  hit= pointer to the rechit
	if(trackdirection.z()!=0){
          // the local angle (single hit)
          float localpitch = static_cast<const StripTopology&>(det->topology()).localPitch(tsos.localPosition());
          float thickness = ((((((hit->geographicalId())>>25)&0x7f)==0xd)||
	                     ((((hit->geographicalId())>>25)&0x7f)==0xe))&&
			           ((((hit->geographicalId())>>5)&0x7)>4)) ? 0.0500 : 0.0320;
          float angle = computeAngleCorr(trackdirection, localpitch, thickness);
	  hitangleassociation.push_back(make_pair(make_pair(hit->geographicalId(),hit->localPosition()), angle)); 
	}
    }
  }
  return hitangleassociation;
}

double SiStripFineDelayTLA::computeAngleCorr(const LocalVector& v, double pitch, double thickness)
{
  double v_xy = sqrt(v.x()*v.x()+v.y()*v.y());
  double L = fabs(thickness*v_xy/v.z());
  double Lmax = fabs(pitch/v.x()*v_xy);
  if(L<Lmax) {
    LogDebug("SiStripFineDelayTLA ") << L << " vs " << Lmax 
        << " Signal contained in strip. Correction is " << v.z()/v.mag();
    return v.z()/v.mag();
  } else {
    LogDebug("SiStripFineDelayTLA ") << L << " vs " << Lmax
       << " Signal not contained in strip. Correction is " << thickness/pitch*v.x()/v_xy*v.z()/v.mag()
       << " instead of " << v.z()/v.mag();
    return thickness/pitch*v.x()/v_xy*v.z()/v.mag();
  }
}

