#ifndef LaserAlignment_SeedGeneratorForLaserBeams_h
#define LaserAlignment_SeedGeneratorForLaserBeams_h

/** \class SeedGeneratorForLaserBeams
 *  seed finding algorithm for the LAS
 *
 *  $Date: 2007/03/18 19:00:19 $
 *  $Revision: 1.2 $
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
