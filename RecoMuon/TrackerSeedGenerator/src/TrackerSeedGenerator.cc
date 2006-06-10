#include "RecoMuon/TrackerSeedGenerator/src/TrackerSeedGenerator.h"

//---------------
// C++ Headers --
//---------------

#include <string>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"
#include "RecoMuon/TrackerSeedGenerator/interface/PrimitiveMuonSeed.h"
#include "RecoMuon/TrackerSeedGenerator/interface/MuonSeedFromConsecutiveHits.h"
#include "RecoTracker/TkSeedGenerator/interface/CombinatorialSeedGeneratorFromPixel.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"


//---------------------------------
//       class TrackerSeedGenerator
//---------------------------------

TrackerSeedGenerator::TrackerSeedGenerator(const MagneticField *field, edm::ParameterSet const& par) :
   thePropagator(new AnalyticalPropagator(field, oppositeToMomentum)),
   theStepPropagator(new SteppingHelixPropagator(field,oppositeToMomentum)),
   theSeedGenerator(new CombinatorialSeedGeneratorFromPixel(par)),
   theVertexPos(GlobalPoint(0.0,0.0,0.0)),
   theVertexErr(GlobalError(0.0001,0.0,0.0001,0.0,0.0,28.09))
{
  edm::LogInfo ("TrackSeedGeneratorFromMuon")<<"TrackerSeedGeneratorFromMuon";
  
  theOption = par.getParameter<int>("SeedOption");
  theDirection = static_cast<ReconstructionDirection>(par.getParameter<int>("Direction")); 
  theUseVertex = par.getParameter<bool>("UseVertex");
  theMaxSeeds = par.getParameter<int>("MaxSeeds");
  theMaxLayers = par.getParameter<int>("MaxLayers");
  theErrorRescale = par.getParameter<double>("ErrorRescaleFactor");

}


//----------------
// Destructor   --
//----------------

TrackerSeedGenerator::~TrackerSeedGenerator() {

  delete theSeedGenerator;
  delete theStepPropagator;
  delete thePropagator;

}


//--------------
// Operations --
//--------------

//
// find tracker seeds
//
BTSeedCollection TrackerSeedGenerator::findSeeds(const reco::Track& muon, const edm::Event& , const edm::EventSetup&) const{
  BTSeedCollection result;
  return result;

}

//
// get ordered list of tracker layers which may be used as seeds
//
void TrackerSeedGenerator::findLayerList(const TrajectoryStateOnSurface& traj) {  
                
  theLayerList.clear();
  // we start from the outer surface of the tracker so it's oppositeToMomentum

}

// primitive seeds
//
void TrackerSeedGenerator::primitiveSeeds(const reco::Track& muon, 
                                                  const TrajectoryStateOnSurface& traj) {

  
}


//
// seeds from consecutive hits
//
void TrackerSeedGenerator::consecutiveHitsSeeds(const reco::Track& muon,
                                                        const TrajectoryStateOnSurface& traj,
                                                        const TrackingRegion& regionOfInterest) {

  if ( theLayerList.size() < 2 ) return;


}


//
// create seeds from consecutive hits
//
void TrackerSeedGenerator::createSeed(const MuonSeedDetLayer& outer,
                                              const MuonSeedDetLayer& inner,
                                              const TrackingRegion& regionOfInterest) {

}


//
// seeds from pixels
//
void TrackerSeedGenerator::pixelSeeds(const reco::Track& muon, 
                                              const TrajectoryStateOnSurface& traj,
                                              const TrackingRegion& regionOfInterest,
                                              float deltaEta, float deltaPhi) {

  
}
