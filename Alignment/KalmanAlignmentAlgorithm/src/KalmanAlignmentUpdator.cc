
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdator.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUserVariables.h"


void KalmanAlignmentUpdator::updateUserVariables( const std::vector< Alignable* > & alignables ) const
{
  std::vector< Alignable* >::const_iterator itAlignable = alignables.begin();

  while ( itAlignable != alignables.end() )
  {
    AlignmentParameters* alignmentParameters = ( *itAlignable )->alignmentParameters();

    if ( alignmentParameters != 0 )
    {
      KalmanAlignmentUserVariables* userVariables =
	dynamic_cast< KalmanAlignmentUserVariables* >( alignmentParameters->userVariables() );

      if ( userVariables != 0 ) userVariables->update();
    }

    itAlignable++;
  }
}


const std::vector< Alignable* >
KalmanAlignmentUpdator::alignablesFromAlignableDets( const std::vector< AlignableDet* > alignableDets,
						     AlignmentParameterStore* store ) const
{
  std::vector< Alignable* > alignables;

  std::vector< AlignableDet* >::const_iterator itAD;
  for ( itAD = alignableDets.begin(); itAD != alignableDets.end(); ++itAD )
  {
    alignables.push_back( store->alignableFromAlignableDet( *itAD ) );
  }

  return alignables;
}


const std::vector< AlignableDet* > 
KalmanAlignmentUpdator::alignableDetsFromHits( TransientTrackingRecHit::ConstRecHitContainer recHits,
					       AlignableNavigator* navigator ) const
{
  std::vector< AlignableDet* > alignableDets;

  TransientTrackingRecHit::ConstRecHitContainer::const_iterator itRecHits;

  for ( itRecHits = recHits.begin(); itRecHits != recHits.end(); ++itRecHits ) {
    AlignableDet* aAlignableDet = navigator->alignableDetFromDetId( ( **itRecHits ).geographicalId() );
    if ( aAlignableDet )
      alignableDets.push_back( aAlignableDet );
    else
      throw cms::Exception( "BadAssociation" ) << "[KalmanAlignmentUpdator::alignableDetsFromHits] find AlignableDet associated to hit!";
  }

  return alignableDets;
}
