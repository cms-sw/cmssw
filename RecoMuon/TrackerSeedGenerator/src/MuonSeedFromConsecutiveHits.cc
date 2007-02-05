#include "RecoMuon/TrackerSeedGenerator/interface/MuonSeedFromConsecutiveHits.h"

/**  \class MuonSeedFromConsecutiveHits
 *
 *   Create muon seed from hits in two
 *   consecutive tracker layers
 *
 *
 *   $Date: 2006/07/27 08:50:20 $
 *   $Revision: 1.4 $
 *
 *   \author   N. Neumeister            Purdue University
 *   \author porting C. Liu             Purdue University
 */

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

//---------------------------------
//       class MuonSeedFromConsecutiveHits
//---------------------------------


//----------------
// Constructors --
//----------------

using namespace std;

MuonSeedFromConsecutiveHits::MuonSeedFromConsecutiveHits(const TransientTrackingRecHit& outerHit,
                                                         const TransientTrackingRecHit& innerHit,
                                                         const PropagationDirection direction,
                                                         const GlobalPoint& vertexPos,
                                                         const GlobalError& vertexErr, const edm::EventSetup& iSetup) :
     status(true), theDirection(direction) {

  construct(outerHit,  
            innerHit,
            vertexPos, 
            vertexErr,
            iSetup);

}


MuonSeedFromConsecutiveHits::~MuonSeedFromConsecutiveHits() { 

}


//
//
//
void MuonSeedFromConsecutiveHits::construct(const TransientTrackingRecHit& outerHit,
	                                    const TransientTrackingRecHit& innerHit,
                                            const GlobalPoint& position,
                                            const GlobalError& error,
                                            const edm::EventSetup& iSetup) {

  typedef TrajectoryStateOnSurface TSOS;
  float mass = 0.1056; //C.L.@@:mass hypothesis
  // make a spiral
  FastHelix helix(outerHit.globalPosition(),innerHit.globalPosition(), position, iSetup);
  
  if ( helix.isValid()) {
  
    AlgebraicSymMatrix C(5,1);
    C[3][3] = error.cxx();
    C[4][4] = error.czz();
    CurvilinearTrajectoryError em(C);
  
    FreeTrajectoryState fts(helix.stateAtVertex().parameters(),em);

    PropagatorWithMaterial thePropagator(alongMomentum, mass);
    KFUpdator              theUpdator;          

    TSOS innerState = thePropagator.propagate(fts,innerHit.det()->surface());

    if ( !innerState.isValid() ) {
      status = false;
      return;
    }

    TSOS innerUpdated = theUpdator.update(innerState,innerHit);

    TSOS outerState = thePropagator.propagate(*innerUpdated.freeTrajectoryState(),
			       outerHit.det()->surface());
                               
    if ( !outerState.isValid() ) {
      status = false;
      return;
    }
    
    TSOS outerUpdated = theUpdator.update(outerState, outerHit);

    theInnerMeas = TrajectoryMeasurement(innerState, innerUpdated, &innerHit);
    theOuterMeas = TrajectoryMeasurement(outerState, outerUpdated, &outerHit);
 
  }  
  else {
    cout << "Error in MuonSeedFromConsecutiveHits: invalid helix" << endl;
    status = false;
  }

}


//
//
//
const FreeTrajectoryState& MuonSeedFromConsecutiveHits::freeTrajectoryState() const {

  if ( theDirection == oppositeToMomentum ) {
    return *theInnerMeas.updatedState().freeState();
  }
  else {  
    return *theOuterMeas.updatedState().freeState();
  }  

}


PTrajectoryStateOnDet MuonSeedFromConsecutiveHits::startingState() const {

  TrajectoryStateTransform tsTransform;

  if ( theDirection == oppositeToMomentum ) {
    return *tsTransform.persistentState(theInnerMeas.updatedState(), theInnerMeas.recHit()->geographicalId().rawId());
  }
  else {
    return *tsTransform.persistentState(theOuterMeas.updatedState(), theOuterMeas.recHit()->geographicalId().rawId());
  }


}


//
//
//
PropagationDirection MuonSeedFromConsecutiveHits::direction() const {

  return theDirection;

}


//
//
//
TrajectorySeed::range MuonSeedFromConsecutiveHits::recHits() const {

  range result; 
  if ( status == false ) return result;
  
  edm::OwnVector<TrackingRecHit> hits;
  hits.reserve(2);

  if ( theDirection == oppositeToMomentum ) {
    hits.push_back(const_cast<TrackingRecHit*>(theOuterMeas.recHit()->hit()));
    hits.push_back(const_cast<TrackingRecHit*>(theInnerMeas.recHit()->hit()));
  }
  else {
    hits.push_back(const_cast<TrackingRecHit*>(theInnerMeas.recHit()->hit()));
    hits.push_back(const_cast<TrackingRecHit*>(theOuterMeas.recHit()->hit()));
  }
 return std::make_pair(hits.begin(),hits.end());
}


//
//
//
vector<TrajectoryMeasurement> MuonSeedFromConsecutiveHits::measurements() const {

  vector<TrajectoryMeasurement> result;
  result.reserve(2);
  
  if ( status == false ) return result;
  
  if ( theDirection == oppositeToMomentum ) {
    result.push_back(theOuterMeas);
    result.push_back(theInnerMeas);
  }
  else {
    result.push_back(theInnerMeas);
    result.push_back(theOuterMeas);  
  } 

  return result;

}


//
//
//
bool MuonSeedFromConsecutiveHits::share(const BasicTrajectorySeed&) const {

  return false;

}


//
//
//
MuonSeedFromConsecutiveHits* MuonSeedFromConsecutiveHits::clone() const { 

  return new MuonSeedFromConsecutiveHits(*this);

}

