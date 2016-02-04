
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdator.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUserVariables.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"


void KalmanAlignmentUpdator::updateUserVariables( const std::vector< Alignable* > & alignables ) const
{
  std::vector< Alignable* > updated;
  std::vector< Alignable* >::const_iterator itAlignable = alignables.begin();

  while ( itAlignable != alignables.end() )
  {
    AlignmentParameters* alignmentParameters = ( *itAlignable )->alignmentParameters();

    if ( std::find(updated.begin(),updated.end(),*itAlignable) == updated.end() && alignmentParameters != 0 )
    {
      KalmanAlignmentUserVariables* userVariables =
	dynamic_cast< KalmanAlignmentUserVariables* >( alignmentParameters->userVariables() );

      if ( userVariables != 0 ) userVariables->update();

      updated.push_back( *itAlignable );
    }

    ++itAlignable;
  }
}


const std::vector< Alignable* >
KalmanAlignmentUpdator::alignablesFromAlignableDets( std::vector< AlignableDetOrUnitPtr >& alignableDets,
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


unsigned int
KalmanAlignmentUpdator::nDifferentAlignables( const std::vector<Alignable*>& ali ) const
{
  std::set< Alignable* > list;
  list.insert( ali.begin(), ali.end() );
  unsigned int ndiff = list.size();
  return ndiff;
}
