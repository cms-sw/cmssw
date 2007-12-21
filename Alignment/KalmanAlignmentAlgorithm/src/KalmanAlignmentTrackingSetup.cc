
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentTrackingSetup.h"

#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"

#include <algorithm>
#include <iostream>

KalmanAlignmentTrackingSetup::KalmanAlignmentTrackingSetup( const std::string& id,
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
							    const bool externalReuseMomentumEstimate ) :
  theId( id ),
  theFitter( fitter->clone() ),
  thePropagator( propagator->clone() ),
  theTrackingSubDetIds( trackingIds ),
  theMinTrackingHits( minTrackingHits ),
  theSortInsideOutFlag( sortInsideOut ),
  theReuseMomentumEstimateFlag( reuseMomentumEstimate ),
  theExternalFitter( externalFitter->clone() ),
  theExternalPropagator( externalPropagator->clone() ),
  theExternalTrackingSubDetIds( externalIds ),
  theMinExternalHits( minExternalHits ),
  theExternalSortInsideOutFlag( externalSortInsideOut ),
  theExternalReuseMomentumEstimateFlag( externalReuseMomentumEstimate )
{}


bool KalmanAlignmentTrackingSetup::useForTracking( const ConstRecHitPointer& recHit ) const
{
  const DetId detId( recHit->det()->geographicalId() );
  const SubDetId subdetId( detId.subdetId() );

  std::vector< SubDetId >::const_iterator itFindSubDetId =
    std::find( theTrackingSubDetIds.begin(), theTrackingSubDetIds.end(), subdetId );

  return ( itFindSubDetId != theTrackingSubDetIds.end() );
}


bool KalmanAlignmentTrackingSetup::useForExternalTracking( const ConstRecHitPointer& recHit ) const
{

  const DetId detId( recHit->det()->geographicalId() );
  const SubDetId subdetId( detId.subdetId() );

  std::vector< SubDetId >::const_iterator itFindSubDetId =
    std::find( theExternalTrackingSubDetIds.begin(), theExternalTrackingSubDetIds.end(), subdetId );

  return ( itFindSubDetId != theExternalTrackingSubDetIds.end() );
}
