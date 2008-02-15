
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
							    const TrajectoryFitter* externalFitter,
							    const Propagator* externalPropagator,
							    const std::vector< SubDetId >& externalIds,
							    const unsigned int minExternalHits,
							    const bool externalSortInsideOut,
							    TrajectoryFactoryBase* trajectoryFactory,
							    KalmanAlignmentUpdator* alignmentUpdator,
							    KalmanAlignmentMetricsUpdator* metricsUpdator ) :
  theId( id ),
  theFitter( fitter->clone() ),
  thePropagator( propagator->clone() ),
  theTrackingSubDetIds( trackingIds ),
  theMinTrackingHits( minTrackingHits ),
  theSortInsideOutFlag( sortInsideOut ),
  theExternalFitter( externalFitter->clone() ),
  theExternalPropagator( externalPropagator->clone() ),
  theExternalTrackingSubDetIds( externalIds ),
  theMinExternalHits( minExternalHits ),
  theExternalSortInsideOutFlag( externalSortInsideOut ),
  theTrajectoryFactory( trajectoryFactory ),
  theAlignmentUpdator( alignmentUpdator ),
  theMetricsUpdator( metricsUpdator )
{}


KalmanAlignmentTrackingSetup::KalmanAlignmentTrackingSetup( const KalmanAlignmentTrackingSetup& setup ) :
  theId( setup.id() ),
  theFitter( setup.fitter()->clone() ),
  thePropagator( setup.propagator()->clone() ),
  theTrackingSubDetIds( setup.getTrackingSubDetIds() ),
  theMinTrackingHits( setup.minTrackingHits() ),
  theSortInsideOutFlag( setup.sortInsideOut() ),
  theExternalFitter( setup.externalFitter()->clone() ),
  theExternalPropagator( setup.externalPropagator()->clone() ),
  theExternalTrackingSubDetIds( setup.getExternalTrackingSubDetIds() ),
  theMinExternalHits( setup.minExternalHits() ),
  theExternalSortInsideOutFlag( setup.externalSortInsideOut() ),
  theTrajectoryFactory( setup.trajectoryFactory() ),
  theAlignmentUpdator( setup.alignmentUpdator() ),
  theMetricsUpdator( setup.metricsUpdator() )
{}


KalmanAlignmentTrackingSetup::~KalmanAlignmentTrackingSetup( void )
{
  if ( theFitter ) delete theFitter;
  if ( theExternalFitter ) delete theExternalFitter;

  if ( thePropagator ) delete thePropagator;
  if ( theExternalPropagator ) delete theExternalPropagator;
}


bool KalmanAlignmentTrackingSetup::useForTracking( const ConstRecHitPointer& recHit ) const
{
  const DetId detId( recHit->det()->geographicalId() );
  const SubDetId subdetId( detId.subdetId() );

//   const GlobalPoint position( recHit->det()->position() );
//   if ( ( position.phi() < 0.785 ) || ( position.phi() > 2.356 ) ) return false;

  std::vector< SubDetId >::const_iterator itFindSubDetId =
    std::find( theTrackingSubDetIds.begin(), theTrackingSubDetIds.end(), subdetId );

//   bool doubleSided = false;
//   if ( subdetId == 3 ) 
//   { 
//     TIBDetId tibid( detId.rawId() ); 
//     if ( tibid.layer() < 3 ) doubleSided = true;
//   }
//   else if ( subdetId == 5 )
//   { 
//     TOBDetId tobid( detId.rawId() ); 
//     if ( tobid.layer() < 3 ) doubleSided = true;
//   }

  return ( itFindSubDetId != theTrackingSubDetIds.end() );// && doubleSided;

}


bool KalmanAlignmentTrackingSetup::useForExternalTracking( const ConstRecHitPointer& recHit ) const
{

  const DetId detId( recHit->det()->geographicalId() );
  const SubDetId subdetId( detId.subdetId() );

//   const GlobalPoint position( recHit->det()->position() );
//   if ( ( position.phi() < 0.785 ) || ( position.phi() > 2.356 ) ) return false;

  std::vector< SubDetId >::const_iterator itFindSubDetId =
    std::find( theExternalTrackingSubDetIds.begin(), theExternalTrackingSubDetIds.end(), subdetId );

//   bool doubleSided = false;
//   if ( subdetId == 3 ) 
//   { 
//     TIBDetId tibid( detId.rawId() ); 
//     if ( tibid.layer() < 3 ) doubleSided = true;
//   }
//   else if ( subdetId == 5 )
//   { 
//     TOBDetId tobid( detId.rawId() ); 
//     if ( tobid.layer() < 3 ) doubleSided = true;
//   }

  return ( itFindSubDetId != theExternalTrackingSubDetIds.end() );// && !doubleSided;
}
