#ifndef _HICDIMUONSEEDGENERATOR_H_
#define _HICDIMUONSEEDGENERATOR_H_
//
// MuSeedGenerator creates track seeds from L1/L2 Muon Trigger
// parametrization works up to 40 GeV tracks.
//
// HIC

// Producer stuff
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

// Magnetic field for propagators
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

// TrackingTools
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"

// Detector geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// RecoTracker
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

// Internal stuff 
#include "RecoHIMuon/HiMuSeed/interface/HICSeedMeasurementEstimator.h"
#include "RecoHIMuon/HiMuSeed/interface/HICConst.h"
#include "RecoHIMuon/HiMuSeed/interface/DiMuonTrajectorySeed.h"

// HICConst
#include "RecoHIMuon/HiMuSeed/interface/HICConst.h"

// CLHEP includes
//#include "CLHEP/Units/GlobalPhysicalConstants.h"
//#include "CLHEP/Vector/ThreeVector.h"
//#include <CLHEP/Vector/LorentzVector.h>

// System includes
#include <cmath>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>

#include <map>
#include <vector>

namespace cms
{
class DiMuonSeedGeneratorHIC  {

 public:
  typedef std::vector<DiMuonTrajectorySeed> SeedContainer;
  typedef SeedContainer::iterator SeedIterator;
  
  DiMuonSeedGeneratorHIC(edm::InputTag,
                         const MagneticField*, 
                         const GeometricSearchTracker*,
                         const HICConst*,
                         const std::string, 
			 int aMult);

  virtual std::map<DetLayer*,SeedContainer>  produce(const edm::Event& e, const edm::EventSetup& c,
  			 FreeTrajectoryState&, 
			 TrajectoryStateOnSurface&,
			 FreeTrajectoryState&,
		    const TransientTrackingRecHitBuilder* RecHitBuilder,
		    const MeasurementTracker* measurementTracker,
			 std::vector<DetLayer*>* 
                       );

  virtual ~DiMuonSeedGeneratorHIC(){} 
  
//  SeedContainer seeds();
  
 private:
  FreeTrajectoryState                        theFtsTracker;
  FreeTrajectoryState                        theFtsMuon;
  const HICConst*                            theHICConst;
  Propagator*                                thePropagator; 
  std::string                                builderName;    
  int                                        theLowMult;
  
  std::vector<BarrelDetLayer*>               bl;
  std::vector<ForwardDetLayer*>              fpos;
  std::vector<ForwardDetLayer*>              fneg;
  std::vector<const DetLayer*>               theDetLayer;
  
  std::vector<LayerWithHits*>                allLayersWithHits;
  bool                                       isFirstCall;
  TrackerLayerIdAccessor                     acc;
  
  std::pair<TrajectoryMeasurement,bool> barrelUpdateSeed(
                                           const FreeTrajectoryState&,
					   const TrajectoryMeasurement&
					 ) const;
  std::pair<TrajectoryMeasurement,bool> forwardUpdateSeed(
                                           const FreeTrajectoryState&,
					   const TrajectoryMeasurement&
					 ) const;
					 
  edm::InputTag rphirecHitsTag;
  const TransientTrackingRecHitBuilder * TTRHbuilder;
  const MagneticField*                   magfield;	
  const GeometricSearchTracker*          theTracker;
  const MeasurementTracker*              theMeasurementTracker;
  const LayerMeasurements*               theLayerMeasurements;
  			 
};
}
#endif

