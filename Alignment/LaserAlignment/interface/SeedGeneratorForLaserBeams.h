/* 
 * seed finding algorithm for the LAS
 */

#ifndef LaserAlignment_SeedGeneratorForLaserBeams_h
#define LaserAlignment_SeedGeneratorForLaserBeams_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"

#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromLayerPairs.h"

#include "Alignment/LaserAlignment/interface/LaserHitPairGenerator.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

class SeedGeneratorForLaserBeams : public SeedGeneratorFromTrackingRegion
{
 public:
  typedef TrajectoryStateOnSurface TSOS;
  
  SeedGeneratorForLaserBeams(const edm::ParameterSet& iConfig);
  virtual ~SeedGeneratorForLaserBeams();
  
  void init(const SiStripRecHit2DCollection &collstereo,
	    const SiStripRecHit2DCollection &collrphi,
	    const SiStripMatchedRecHit2DCollection &collmatched,
	    const edm::EventSetup & iSetup);

  void run(TrajectorySeedCollection &, const edm::EventSetup & iSetup);

 private:
  edm::ParameterSet conf_;
  GlobalTrackingRegion region;
  LaserHitPairGenerator * thePairGenerator;
  edm::ESHandle<MagneticField> magfield;
  edm::ESHandle<TrackerGeometry> tracker;
  TrajectoryStateTransform transformer;
  KFUpdator * theUpdator;
  PropagatorWithMaterial * thePropagatorAl;
  PropagatorWithMaterial * thePropagatorOp;
  const TransientTrackingRecHitBuilder * TTRHBuilder;
  std::string builderName;
};
#endif
