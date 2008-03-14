/** \file CosmicMuonUtilities
 *
 *
 *  $Date: 2007/03/08 20:25:28 $
 *  $Revision: 1.1 $
 *  \author Chang Liu  -  Purdue University
 */

#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonUtilities.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"

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

TrajectoryStateOnSurface CosmicMuonUtilities::stepPropagate(const TrajectoryStateOnSurface& tsos,
                                              const ConstRecHitPointer& hit,
                                              const Propagator& prop ) const {

  const std::string metname = "Muon|RecoMuon|CosmicMuonUtilities";

  GlobalPoint start = tsos.globalPosition();
  GlobalPoint dest = hit->globalPosition();
  GlobalVector StepVector = dest - start;
  GlobalVector UnitStepVector = StepVector.unit();
  GlobalPoint GP =start;
  TrajectoryStateOnSurface result(tsos);
  float totalDis = StepVector.mag();
  LogDebug(metname)<<"stepPropagate: propagate from: "<<start<<" to "<<dest;
  LogDebug(metname)<<"stepPropagate: their distance: "<<totalDis;

  int steps = 3; // need to optimize

  float oneStep = totalDis/steps;
  Basic3DVector<float> Basic3DV(StepVector.x(),StepVector.y(),StepVector.z());
  for ( int istep = 0 ; istep < steps - 1 ; istep++) {
        GP += oneStep*UnitStepVector;
        Surface::PositionType pos(GP.x(),GP.y(),GP.z());
        LogDebug(metname)<<"stepPropagate: a middle plane: "<<pos<<endl;
        Surface::RotationType rot( Basic3DV , float(0));
        PlaneBuilder::ReturnType SteppingPlane = PlaneBuilder().plane(pos,rot);
        TrajectoryStateOnSurface predTsos = prop.propagate( result, *SteppingPlane);
        if (predTsos.isValid()) {
            result=predTsos;
            LogDebug(metname)<<"result "<< result.globalPosition()<<endl;
          }
 }

  TrajectoryStateOnSurface predTsos = prop.propagate( result, hit->det()->surface());
  if (predTsos.isValid()) result=predTsos;

  return result;
}


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
