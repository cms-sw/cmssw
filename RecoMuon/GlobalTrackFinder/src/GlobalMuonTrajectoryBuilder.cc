/**
 *  Class: GlobalMuonTrajectoryBuilder
 *
 *  Description: 
 *
 *             MuonHitsOption: 0 - tracker only
 *                             1 - include all muon hits
 *                             2 - include only first muon hit(s)
 *                             3 - include only selected muon hits
 *                             4 - combined
 *
 *
 *  $Date: 2006/07/12 04:26:39 $
 *  $Revision: 1.5 $
 *
 *  Author :
 *  N. Neumeister            Purdue University
 *  with contributions from: S. Lacaprara, J. Mumford, P. Traczyk
 *  porting author:
 *  C. Liu                   Purdue University
 *
 **/

#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
//#include "RecoMuon/TrackerSeedGenerator/src/TrackerSeedGenerator.h"
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonSeedCleaner.h"
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonReFitter.h"
#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "RecoMuon/GlobalMuonProducer/src/GlobalMuonProducer.h"

using namespace std;
using namespace edm;
//----------------
// Constructors --
//----------------

GlobalMuonTrajectoryBuilder::GlobalMuonTrajectoryBuilder(const edm::ParameterSet& par) 
{

}


//--------------
// Destructor --
//--------------

GlobalMuonTrajectoryBuilder::~GlobalMuonTrajectoryBuilder() 
{

}

MuonTrajectoryBuilder::CandidateContainer GlobalMuonTrajectoryBuilder::trajectories(const reco::Track&)
{
  MuonTrajectoryBuilder::CandidateContainer result;
  return result;
}

void GlobalMuonTrajectoryBuilder::setES(const edm::EventSetup& setup){

}

void GlobalMuonTrajectoryBuilder::setEvent(const edm::Event& event)
{

}
