//#include "Alignment/KalmanAlignmentAlgorithm/plugins/DummyMetricsUpdator.h"
#include "DummyMetricsUpdator.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdatorPlugin.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"


DummyMetricsUpdator::DummyMetricsUpdator( const edm::ParameterSet & config ) : KalmanAlignmentMetricsUpdator( config )
{
  std::vector< unsigned int > dummy;
  theFixedAlignableIds = config.getUntrackedParameter< std::vector<unsigned int> >( "FixedAlignableIds", dummy );
}


void DummyMetricsUpdator::update( const std::vector< Alignable* > & alignables )
{
  std::vector< Alignable* >::const_iterator itAD = alignables.begin();
  while ( itAD != alignables.end() )
  {
    unsigned int subdetId = static_cast< unsigned int >( (*itAD)->geomDetId().subdetId() );
    if ( find( theFixedAlignableIds.begin(), theFixedAlignableIds.end(), subdetId ) == theFixedAlignableIds.end() )
    {
      theSetOfAllAlignables.insert( *itAD );
    }
    ++itAD;
  }
}


const std::vector< Alignable* >
DummyMetricsUpdator::additionalAlignables( const std::vector< Alignable* > & alignables )
{
  std::vector< Alignable* > result;
  result.reserve( theSetOfAllAlignables.size() );

  std::set< Alignable* >::iterator itS = theSetOfAllAlignables.begin();
  while ( itS != theSetOfAllAlignables.end() )
  {
    if ( find( alignables.begin(), alignables.end(), *itS ) == alignables.end() ) result.push_back( *itS );
    ++itS;
  }

  return result;
}


const std::map< Alignable*, short int >
DummyMetricsUpdator::additionalAlignablesWithDistances( const std::vector< Alignable* > & alignables )
{
  std::map< Alignable*, short int > result;

  std::set< Alignable* >::iterator itS = theSetOfAllAlignables.begin();
  while ( itS != theSetOfAllAlignables.end() )
  {
    if ( find( alignables.begin(), alignables.end(), *itS ) == alignables.end() ) result[*itS] = 0;
    ++itS;
  }

  return result;
}


const std::vector< Alignable* > DummyMetricsUpdator::alignables( void ) const
{
  std::vector< Alignable* > alignables;
  alignables.reserve( theSetOfAllAlignables.size() );
  alignables.insert( alignables.begin(), theSetOfAllAlignables.begin(), theSetOfAllAlignables.end() );
  return alignables;
}


DEFINE_EDM_PLUGIN( KalmanAlignmentMetricsUpdatorPlugin, DummyMetricsUpdator, "DummyMetricsUpdator" );
