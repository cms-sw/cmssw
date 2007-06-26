#ifndef LaserAlignment_SeedGeneratorForLaserBeams_h
#define LaserAlignment_SeedGeneratorForLaserBeams_h

/** \class SeedGeneratorForLaserBeams
 *  seed finding algorithm for the LAS
 *
 *  $Date: 2007/05/10 12:00:32 $
 *  $Revision: 1.3 $
 *  \author Maarten Thomas
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"

#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "Alignment/LaserAlignment/interface/LaserHitPairGenerator.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

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
  void propagateWithMaterial(edm::OwnVector<TrackingRecHit> & hits, OrderedLaserHitPairs & HitPairs, GlobalPoint & inner, GlobalPoint & outer);
  /// propagate using AnalyticalPropagator
  void propagateAnalytical(edm::OwnVector<TrackingRecHit> & hits, OrderedLaserHitPairs & HitPairs, GlobalPoint & inner, GlobalPoint & outer);
   
  edm::ParameterSet conf_;
  GlobalTrackingRegion region;
  LaserHitPairGenerator * thePairGenerator;
  edm::ESHandle<MagneticField> magfield;
  edm::ESHandle<TrackerGeometry> tracker;
  TrajectoryStateTransform transformer;
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
