/** \file CosmicMuonUtilities
 *
 *
 *  $Date: 2010/03/07 21:28:57 $
 *  $Revision: 1.6 $
 *  \author Chang Liu  -  Purdue University
 */

#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonUtilities.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

//
// constructor
//
CosmicMuonUtilities::CosmicMuonUtilities() {
}


//
// destructor
//
CosmicMuonUtilities::~CosmicMuonUtilities() {
}


void CosmicMuonUtilities::reverseDirection(TrajectoryStateOnSurface& tsos, const MagneticField* mgfield) const {

   GlobalTrajectoryParameters gtp(tsos.globalPosition(),
                                  -tsos.globalMomentum(),
                                  -tsos.charge(),
                                  mgfield);
   TrajectoryStateOnSurface newTsos(gtp, tsos.cartesianError(), tsos.surface()); 
   tsos = newTsos;

   return;

}


//
//
//
string CosmicMuonUtilities::print(const MuonTransientTrackingRecHit::ConstMuonRecHitContainer& hits) const {

   stringstream output;

    for (ConstMuonRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      output << "invalid RecHit"<<endl;
      continue;
    }

    const GlobalPoint& pos = (*ir)->globalPosition();
    output
    << "pos"<<pos
    << "radius "<<pos.perp()
    << "  dim " << (*ir)->dimension()
    << "  det " << (*ir)->det()->geographicalId().det()
    << "  sub det " << (*ir)->det()->subDetector()<<endl;
  }

  return output.str();

}


//
//
//
string CosmicMuonUtilities::print(const MuonTransientTrackingRecHit::MuonRecHitContainer& hits) const {

   stringstream output;

    for (MuonRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      output << "invalid RecHit"<<endl;
      continue;
    }

    const GlobalPoint& pos = (*ir)->globalPosition();
    output
    << "pos"<<pos
    << "radius "<<pos.perp()
    << "  dim " << (*ir)->dimension()
    << "  det " << (*ir)->det()->geographicalId().det()
    << "  sub det " << (*ir)->det()->subDetector()<<endl;
  }

  return output.str();

}


//
//
//
string CosmicMuonUtilities::print(const TransientTrackingRecHit::ConstRecHitContainer& hits) const {

    stringstream output;

    for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      output << "invalid RecHit"<<endl;
      continue;
    }

    const GlobalPoint& pos = (*ir)->globalPosition();
    output
    << "pos"<<pos
    << "radius "<<pos.perp()
    << "  dim " << (*ir)->dimension()
    << "  det " << (*ir)->det()->geographicalId().det()
    << "  sub det " << (*ir)->det()->subDetector()<<endl;
  }

  return output.str();

}


//
//
//
bool CosmicMuonUtilities::isTraversing(const Trajectory& t) const {

  TransientTrackingRecHit::ConstRecHitContainer hits = t.recHits();

  if ( hits.empty() ) return false;

  ConstRecHitContainer::const_iterator frontHit = hits.begin();
  ConstRecHitContainer::const_iterator backHit  = hits.end() - 1;

  // find first valid hit at both ends of the trajectory
  while ( !(*frontHit)->isValid() && frontHit != backHit) {++frontHit;}
  while ( !(*backHit)->isValid() && backHit != frontHit)  {--backHit;}

  if ( frontHit == backHit )  return false;

  GlobalPoint frontPos = (*frontHit)->globalPosition();
  GlobalPoint backPos = (*backHit)->globalPosition();

  // are there separate muon hits in 2 different hemispheres
  GlobalVector deltaPos(frontPos - backPos);
  if ( deltaPos.mag() < 100.0 ) return false;
  if ( fabs(deltaPos.z() ) > 500.0 ) return true;
  if ( deltaPos.perp() > 350.0 ) return true;
  GlobalPoint middle((frontPos.x()+backPos.x())/2, 
                     (frontPos.y()+backPos.y())/2, 
                     (frontPos.z()+backPos.z())/2);

  return ( (middle.perp() < frontPos.perp()) && (middle.perp() < backPos.perp()) );

}


//
//
//
TrajectoryStateOnSurface CosmicMuonUtilities::stepPropagate(const TrajectoryStateOnSurface& tsos,
                                                            const ConstRecHitPointer& hit,
                                                            const Propagator& prop ) const {

  const std::string metname = "Muon|RecoMuon|CosmicMuonUtilities";

  GlobalPoint start = tsos.globalPosition();
  GlobalPoint dest = hit->globalPosition();
  GlobalVector StepVector = dest - start;
  GlobalVector UnitStepVector = StepVector.unit();
  GlobalPoint GP =start;
  TrajectoryStateOnSurface currTsos(tsos);
  TrajectoryStateOnSurface predTsos;
  float totalDis = StepVector.mag();
  LogTrace(metname)<<"stepPropagate: propagate from: "<<start<<" to "<<dest;
  LogTrace(metname)<<"stepPropagate: their distance: "<<totalDis;

  int steps = 3; // need to optimize

  float oneStep = totalDis/steps;
  Basic3DVector<float> Basic3DV(StepVector.x(),StepVector.y(),StepVector.z());
  for ( int istep = 0 ; istep < steps - 1 ; istep++) {
        GP += oneStep*UnitStepVector;
        Surface::PositionType pos(GP.x(),GP.y(),GP.z());
        LogTrace(metname)<<"stepPropagate: a middle plane: "<<pos<<endl;
        Surface::RotationType rot( Basic3DV , float(0));
        PlaneBuilder::ReturnType SteppingPlane = PlaneBuilder().plane(pos,rot);
        TrajectoryStateOnSurface predTsos = prop.propagate(currTsos, *SteppingPlane);
        if (predTsos.isValid()) {
            currTsos=predTsos;
            LogTrace(metname)<<"stepPropagate: middle state "<< currTsos.globalPosition()<<endl;
        }
 }

  predTsos = prop.propagate(currTsos, hit->det()->surface());

  return predTsos;

}
