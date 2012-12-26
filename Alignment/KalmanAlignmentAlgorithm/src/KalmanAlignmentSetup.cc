
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentSetup.h"

//#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
//#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <algorithm>
#include <iostream>

KalmanAlignmentSetup::KalmanAlignmentSetup( const std::string& id,
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
							    KalmanAlignmentMetricsUpdator* metricsUpdator ) :
  theId( id ),
  theFitter( fitter->clone() ),
  thePropagator( propagator->clone() ),
  theTrackingSubDetIds( trackingIds ),
  theMinTrackingHits( minTrackingHits ),
  theSortingDir( sortingDir ),
  theExternalFitter( externalFitter->clone() ),
  theExternalPropagator( externalPropagator->clone() ),
  theExternalTrackingSubDetIds( externalIds ),
  theMinExternalHits( minExternalHits ),
  theExternalSortingDir( externalSortingDir ),
  theTrajectoryFactory( trajectoryFactory ),
  theAlignmentUpdator( alignmentUpdator ),
  theMetricsUpdator( metricsUpdator )
{}


KalmanAlignmentSetup::KalmanAlignmentSetup( const KalmanAlignmentSetup& setup ) :
  theId( setup.id() ),
  theFitter( setup.fitter()->clone() ),
  thePropagator( setup.propagator()->clone() ),
  theTrackingSubDetIds( setup.getTrackingSubDetIds() ),
  theMinTrackingHits( setup.minTrackingHits() ),
  theSortingDir( setup.sortingDirection() ),
  theExternalFitter( setup.externalFitter()->clone() ),
  theExternalPropagator( setup.externalPropagator()->clone() ),
  theExternalTrackingSubDetIds( setup.getExternalTrackingSubDetIds() ),
  theMinExternalHits( setup.minExternalHits() ),
  theExternalSortingDir( setup.externalSortingDirection() ),
  theTrajectoryFactory( setup.trajectoryFactory() ),
  theAlignmentUpdator( setup.alignmentUpdator() ),
  theMetricsUpdator( setup.metricsUpdator() )
{}


KalmanAlignmentSetup::~KalmanAlignmentSetup( void )
{
  if ( theFitter ) delete theFitter;
  if ( theExternalFitter ) delete theExternalFitter;

  if ( thePropagator ) delete thePropagator;
  if ( theExternalPropagator ) delete theExternalPropagator;
}


bool KalmanAlignmentSetup::useForTracking( const ConstRecHitPointer& recHit ) const
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
//      
//     if ( tTopo->tibLayer( detId.rawId) < 3 ) doubleSided = true;
//   }
//   else if ( subdetId == 5 )
//   { 
//      
//     if ( tTopo->tobLayer( detId.rawId) < 3 ) doubleSided = true;
//   }

  return ( itFindSubDetId != theTrackingSubDetIds.end() );// && doubleSided;

}


bool KalmanAlignmentSetup::useForExternalTracking( const ConstRecHitPointer& recHit ) const
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
//      
//     if ( tTopo->tibLayer( detId.rawId) < 3 ) doubleSided = true;
//   }
//   else if ( subdetId == 5 )
//   { 
//      
//     if ( tTopo->tobLayer( detId.rawId) < 3 ) doubleSided = true;
//   }

  return ( itFindSubDetId != theExternalTrackingSubDetIds.end() );// && !doubleSided;
}
