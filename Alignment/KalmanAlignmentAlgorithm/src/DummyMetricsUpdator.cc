#include "Alignment/KalmanAlignmentAlgorithm/interface/DummyMetricsUpdator.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"


DummyMetricsUpdator::DummyMetricsUpdator( const edm::ParameterSet & config ) : KalmanAlignmentMetricsUpdator( config )
{
  std::vector< unsigned int > dummy;
  theFixedAlignableDetIds = config.getUntrackedParameter< std::vector<unsigned int> >( "FixedAlignableDetIds", dummy );
}


void DummyMetricsUpdator::update( const std::vector< AlignableDet* > & alignableDets )
{
  std::vector< AlignableDet* >::const_iterator itAD = alignableDets.begin();
  while ( itAD != alignableDets.end() )
  {
    unsigned int subdetId = static_cast< unsigned int >( (*itAD)->geomDetId().subdetId() );
    if ( find( theFixedAlignableDetIds.begin(), theFixedAlignableDetIds.end(), subdetId ) == theFixedAlignableDetIds.end() )
    {
      theSetOfAllAlignableDets.insert( *itAD );
    }
    ++itAD;
  }
}


const std::vector< AlignableDet* >
DummyMetricsUpdator::additionalAlignableDets( const std::vector< AlignableDet* > & alignableDets )
{
  std::vector< AlignableDet* > result;
  result.reserve( theSetOfAllAlignableDets.size() );

  std::set< AlignableDet* >::iterator itS = theSetOfAllAlignableDets.begin();
  while ( itS != theSetOfAllAlignableDets.end() )
  {
    if ( find( alignableDets.begin(), alignableDets.end(), *itS ) == alignableDets.end() ) result.push_back( *itS );
    ++itS;
  }

  return result;
}


const std::map< AlignableDet*, short int >
DummyMetricsUpdator::additionalAlignableDetsWithDistances( const std::vector< AlignableDet* > & alignableDets )
{
  std::map< AlignableDet*, short int > result;

  std::set< AlignableDet* >::iterator itS = theSetOfAllAlignableDets.begin();
  while ( itS != theSetOfAllAlignableDets.end() )
  {
    if ( find( alignableDets.begin(), alignableDets.end(), *itS ) == alignableDets.end() ) result[*itS] = 0;
    ++itS;
  }

  return result;
}
