#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentTrackingSetup_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentTrackingSetup_h

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

#include <vector>


class KalmanAlignmentTrackingSetup
{

public:

  typedef int SubDetId;

  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

  KalmanAlignmentTrackingSetup( const std::string& id,
				const TrajectoryFitter* fitter,
				const Propagator* propagator,
				const std::vector< SubDetId >& trackingIds,
				const unsigned int minTrackingHits,
				const bool sortInsideOut,
				const bool reuseMomentumEstimate,
				const TrajectoryFitter* externalFitter,
				const Propagator* externalPropagator,
				const std::vector< SubDetId >& externalIds,
				const unsigned int minExternalHits,
				const bool externalSortInsideOut,
				const bool externalReuseMomentumEstimate );

  ~KalmanAlignmentTrackingSetup( void ) {}

  inline const std::string id( void ) const { return theId; }

  inline const TrajectoryFitter* fitter( void ) const { return theFitter.operator->(); }
  inline const TrajectoryFitter* externalFitter( void ) const { return theExternalFitter.operator->(); }

  inline const Propagator* propagator( void ) const { return thePropagator.operator->(); }
  inline const Propagator* externalPropagator( void ) const { return theExternalPropagator.operator->(); }

  inline const std::vector< SubDetId >& getTrackingSubDetIds( void ) const { return theTrackingSubDetIds; }
  inline const std::vector< SubDetId >& getExternalTrackingSubDetIds( void ) const { return theExternalTrackingSubDetIds; }

  inline const unsigned int minTrackingHits( void ) const { return theMinTrackingHits; }
  inline const unsigned int minExternalHits( void ) const { return theMinExternalHits; }

  inline const bool sortInsideOut( void ) const { return theSortInsideOutFlag; }
  inline const bool externalSortInsideOut( void ) const { return theExternalSortInsideOutFlag; }

  inline const bool reuseMomentumEstimate( void ) const { return theReuseMomentumEstimateFlag; }
  inline const bool externalReuseMomentumEstimate( void ) const { return theExternalReuseMomentumEstimateFlag; }

  bool useForTracking( const ConstRecHitPointer& recHit ) const;
  bool useForExternalTracking( const ConstRecHitPointer& recHit ) const;


private:

  std::string theId;

  DeepCopyPointerByClone< const TrajectoryFitter > theFitter;
  DeepCopyPointerByClone< const Propagator > thePropagator;
  std::vector< SubDetId > theTrackingSubDetIds;
  unsigned int theMinTrackingHits;
  bool theSortInsideOutFlag;
  bool theReuseMomentumEstimateFlag;

  DeepCopyPointerByClone< const TrajectoryFitter > theExternalFitter;
  DeepCopyPointerByClone< const Propagator > theExternalPropagator;
  std::vector< SubDetId > theExternalTrackingSubDetIds;
  unsigned int theMinExternalHits;
  bool theExternalSortInsideOutFlag;
  bool theExternalReuseMomentumEstimateFlag;
};

#endif
