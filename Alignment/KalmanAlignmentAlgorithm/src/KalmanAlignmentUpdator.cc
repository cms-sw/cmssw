
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
KalmanAlignmentUpdator::alignablesFromAlignableDets( const std::vector< AlignableDetOrUnitPtr >& alignableDets,
						     AlignmentParameterStore* store ) const
{
  std::vector< Alignable* > alignables;

  std::vector< AlignableDetOrUnitPtr >::const_iterator itAD;
  for ( itAD = alignableDets.begin(); itAD != alignableDets.end(); ++itAD )
  {
    Alignable* ali = store->alignableFromAlignableDet( *itAD );
    alignables.push_back( ali );
  }

  return alignables;
}
