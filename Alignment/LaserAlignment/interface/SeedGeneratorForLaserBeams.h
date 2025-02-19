#ifndef LaserAlignment_SeedGeneratorForLaserBeams_h
#define LaserAlignment_SeedGeneratorForLaserBeams_h

/** \class SeedGeneratorForLaserBeams
 *  seed finding algorithm for the LAS
 *
 *  $Date: 2011/12/22 18:06:57 $
 *  $Revision: 1.8 $
 *  \author Maarten Thomas
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"

#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "Alignment/LaserAlignment/interface/LaserHitPairGenerator.h"
#include "DataFormats/Common/interface/RangeMap.h" 
#include "Alignment/LaserAlignment/interface/LaserHitPairGeneratorFromLayerPair.h" 


#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

class SeedGeneratorForLaserBeams
{
 public:
  typedef TrajectoryStateOnSurface TSOS;
  
	/// constructor
  SeedGeneratorForLaserBeams(const edm::ParameterSet& iConfig);
	/// destructor
  virtual ~SeedGeneratorForLaserBeams();
  
	/// initialize seed finder algorithm
  void init(const SiStripRecHit2DCollection &collstereo,
	    const SiStripRecHit2DCollection &collrphi,
	    const SiStripMatchedRecHit2DCollection &collmatched,
	    const edm::EventSetup & iSetup);

	/// run the seed finder
  void run(TrajectorySeedCollection &, const edm::EventSetup & iSetup);

 private:
  /// propagate using PropagatorWithMaterial
  void propagateWithMaterial(OrderedLaserHitPairs & HitPairs, TrajectorySeedCollection & output);
  /// propagate using AnalyticalPropagator
  void propagateAnalytical(OrderedLaserHitPairs & HitPairs, TrajectorySeedCollection & output);
   
  edm::ParameterSet conf_;
  GlobalTrackingRegion region;
  LaserHitPairGenerator * thePairGenerator;
  edm::ESHandle<MagneticField> magfield;
  edm::ESHandle<TrackerGeometry> tracker;
  
  KFUpdator * theUpdator;
  PropagatorWithMaterial * thePropagatorMaterialAl;
  PropagatorWithMaterial * thePropagatorMaterialOp;
  AnalyticalPropagator * thePropagatorAnalyticalAl;
  AnalyticalPropagator * thePropagatorAnalyticalOp;
  const TransientTrackingRecHitBuilder * TTRHBuilder;
  std::string builderName;
  std::string propagatorName;
};
#endif
