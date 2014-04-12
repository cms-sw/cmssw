#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentSetup_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentSetup_h

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdator.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdator.h"

#include <vector>
#include <string>


class KalmanAlignmentSetup
{

public:

  typedef int SubDetId;
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

  enum SortingDirection { sortInsideOut, sortOutsideIn, sortUpsideDown, sortDownsideUp };

  KalmanAlignmentSetup( const std::string& id,
			const TrajectoryFitter* fitter,
			const Propagator* propagator,
			const std::vector< SubDetId >& trackingIds,
			const unsigned int minTrackingHits,
			const SortingDirection sortingDir,
			const TrajectoryFitter* externalFitter,
			const Propagator* externalPropagator,
			const std::vector< SubDetId >& externalIds,
			const unsigned int minExternalHits,
			const SortingDirection externalSortingDir,
			TrajectoryFactoryBase* trajectoryFactory,
			KalmanAlignmentUpdator* alignmentUpdator,
			KalmanAlignmentMetricsUpdator* metricsUpdator );

  KalmanAlignmentSetup( const KalmanAlignmentSetup& setup );

  ~KalmanAlignmentSetup( void );

  inline const std::string id( void ) const { return theId; }

  inline const TrajectoryFitter* fitter( void ) const { return theFitter.get(); }
  inline const TrajectoryFitter* externalFitter( void ) const { return theExternalFitter.get(); }

  inline const Propagator* propagator( void ) const { return thePropagator; }
  inline const Propagator* externalPropagator( void ) const { return theExternalPropagator; }

  inline const std::vector< SubDetId >& getTrackingSubDetIds( void ) const { return theTrackingSubDetIds; }
  inline const std::vector< SubDetId >& getExternalTrackingSubDetIds( void ) const { return theExternalTrackingSubDetIds; }

  inline const unsigned int minTrackingHits( void ) const { return theMinTrackingHits; }
  inline const unsigned int minExternalHits( void ) const { return theMinExternalHits; }

  inline const SortingDirection sortingDirection( void ) const { return theSortingDir; }
  inline const SortingDirection externalSortingDirection( void ) const { return theExternalSortingDir; }

  bool useForTracking( const ConstRecHitPointer& recHit ) const;
  bool useForExternalTracking( const ConstRecHitPointer& recHit ) const;

  TrajectoryFactoryBase* trajectoryFactory( void ) const { return theTrajectoryFactory; }
  KalmanAlignmentUpdator* alignmentUpdator( void ) const { return theAlignmentUpdator; }
  KalmanAlignmentMetricsUpdator* metricsUpdator( void ) const { return theMetricsUpdator; }

private:

  std::string theId;

  std::unique_ptr<TrajectoryFitter> theFitter;
  Propagator* thePropagator;
  std::vector< SubDetId > theTrackingSubDetIds;
  unsigned int theMinTrackingHits;
  SortingDirection theSortingDir;

  std::unique_ptr<TrajectoryFitter> theExternalFitter;
  Propagator* theExternalPropagator;
  std::vector< SubDetId > theExternalTrackingSubDetIds;
  unsigned int theMinExternalHits;
  SortingDirection theExternalSortingDir;

  TrajectoryFactoryBase* theTrajectoryFactory;
  KalmanAlignmentUpdator* theAlignmentUpdator;
  KalmanAlignmentMetricsUpdator* theMetricsUpdator;

};

#endif
