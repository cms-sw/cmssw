#include "CalibTracker/SiStripHitEfficiency/interface/TrajectoryAtInvalidHit.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "RecoTracker/MeasurementDet/interface/RecHitPropagator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"

using namespace std;
TrajectoryAtInvalidHit::TrajectoryAtInvalidHit( const TrajectoryMeasurement& tm, 
					    const TrackerGeometry* tracker,
					    const Propagator& propagator,
					    const uint mono)
{
  theCombinedPredictedState = TrajectoryStateCombiner().combine( tm.forwardPredictedState(),
  								 tm.backwardPredictedState());

  if (!theCombinedPredictedState.isValid()) {
    cout << "found invalid combinedpredictedstate"<< endl;
    return;
  }

  theHit = tm.recHit();  
  iidd = theHit->geographicalId().rawId();
  StripSubdetector strip=StripSubdetector(iidd);
  unsigned int subid=strip.subdetId();
  // xB and yB are for absolute borders on the trajectories included in the study, sigmaX sigmaY are 
  // significance cuts on the distance from the detector surface
  float xB = 0.; float sigmaX = 5.0;
  float yB = 0.; float sigmaY = 5.0;
  float sigmaYBond = 0.;
  //set bounds for point to be within to be counted in the study
  if (subid ==  StripSubdetector::TOB) { 
    sigmaYBond = 5.0;
  }
  const GeomDetUnit * monodet;
  
  // if module is from a double sided layer, write out info for either the
  // rphi surface (mono = 1) or the stereo surface (mono = 2)--not the matched hit surface
  if (( mono > 0 ) && isDoubleSided(iidd) ) {
    // find matched det id, that is the matched hit surface between the two sensors
    uint matched_iidd = iidd-(iidd & 0x3);
    DetId matched_id(matched_iidd);
    
    GluedGeomDet * gdet=(GluedGeomDet *)tracker->idToDet(matched_id);
    
    // get the sensor det indicated by mono
    if (mono == 1) monodet=gdet->stereoDet();
    else  monodet=gdet->monoDet(); // this should only be mono == 2

    // set theCombinedPredictedState to be on the sensor surface, not the matched surface
    DetId mono_id = monodet->geographicalId();
    const Surface &surface = tracker->idToDet(mono_id)->surface();
    theCombinedPredictedState = propagator.propagate(theCombinedPredictedState, 
						     surface);

    if (!theCombinedPredictedState.isValid()) {
      cout << "found invalid combinedpredictedstate after propagation"<< endl;
      return;
    }
    
    //set module id to be mono det
    iidd = monodet->geographicalId().rawId();
  } else {
    monodet = (GeomDetUnit*)theHit->det();
  }

  locX = theCombinedPredictedState.localPosition().x();
  locY = theCombinedPredictedState.localPosition().y();
  locZ = theCombinedPredictedState.localPosition().z();
  locXError = sqrt(theCombinedPredictedState.localError().positionError().xx());
  locYError = sqrt(theCombinedPredictedState.localError().positionError().yy());
  locDxDz = theCombinedPredictedState.localParameters().vector()[1];
  locDyDz = theCombinedPredictedState.localParameters().vector()[2];
  globX = theCombinedPredictedState.globalPosition().x();
  globY = theCombinedPredictedState.globalPosition().y();
  globZ = theCombinedPredictedState.globalPosition().z();
  
  // this should never be a glued det, only rphi or stero
  //cout << "From TrajAtValidHit module " << iidd << "   matched/stereo/rphi = " << ((iidd & 0x3)==0) << "/" << ((iidd & 0x3)==1) << "/" << ((iidd & 0x3)==2) << endl;
    
  // Restrict the bound regions for better understanding of the modul assignment. 

  LocalPoint BoundedPoint;
  float xx, yy ,zz;

  // Insert the bounded values 
  if (locX < 0. ) xx = min(locX - xB,locX - sigmaX*locXError);
  else  xx = max(locX + xB, locX + sigmaX*locXError);

  if (locY < 0. ) yy = min(locY - yB,locY - sigmaY*locYError);
  else  yy = max(locY + yB, locY + sigmaY*locYError);

  zz = theCombinedPredictedState.localPosition().z();

  BoundedPoint = LocalPoint(xx,yy,zz);
  
  if ( monodet->surface().bounds().inside(BoundedPoint) && abs(locY) > sigmaYBond*locYError ){
    acceptance = true;
  }
  else {
    // hit is within xB, yB from the edge of the detector, so throw it out 
    acceptance = false;
  }
}

double TrajectoryAtInvalidHit::localX() const
{
  return locX;
}
double TrajectoryAtInvalidHit::localY() const
{
  return locY;
}
double TrajectoryAtInvalidHit::localZ() const
{
  return locZ;
}
double TrajectoryAtInvalidHit::localErrorX() const
{
  return locXError;
}
double TrajectoryAtInvalidHit::localErrorY() const
{
  return locYError;
}
double TrajectoryAtInvalidHit::localDxDz() const {
  return locDxDz;
}
double TrajectoryAtInvalidHit::localDyDz() const {
  return locDyDz;
}
double TrajectoryAtInvalidHit::globalX() const
{
  return globX;
}
double TrajectoryAtInvalidHit::globalY() const
{
  return globY;
}
double TrajectoryAtInvalidHit::globalZ() const
{
  return globZ;
}

uint TrajectoryAtInvalidHit::monodet_id() const
{
  return iidd;
}

bool TrajectoryAtInvalidHit::withinAcceptance() const
{
  return acceptance;
}

bool TrajectoryAtInvalidHit::isDoubleSided(uint iidd) const {
  StripSubdetector strip=StripSubdetector(iidd);
  unsigned int subid=strip.subdetId();
  uint layer = 0;
  if (subid ==  StripSubdetector::TIB) { 
    TIBDetId tibid(iidd);
    layer = tibid.layer();
    if (layer == 1 || layer == 2) return true;
    else return false;
  }
  else if (subid ==  StripSubdetector::TOB) { 
    TOBDetId tobid(iidd);
    layer = tobid.layer() + 4 ; 
    if (layer == 5 || layer == 6) return true;
    else return false;
  }
  else if (subid ==  StripSubdetector::TID) { 
    TIDDetId tidid(iidd);
    layer = tidid.ring() + 10;
    if (layer == 11 || layer == 12) return true;
    else return false;
  }
  else if (subid ==  StripSubdetector::TEC) { 
    TECDetId tecid(iidd);
    layer = tecid.ring() + 13 ; 
    if (layer == 14 || layer == 15 || layer == 18) return true;
    else return false;
  }
  else
    return false;
}

TrajectoryStateOnSurface TrajectoryAtInvalidHit::tsos() const {
  return theCombinedPredictedState;
}
