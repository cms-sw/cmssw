#include "Alignment/KalmanAlignmentAlgorithm/interface/SimpleMetricsUpdator.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"



SimpleMetricsUpdator::SimpleMetricsUpdator( const edm::ParameterSet & config ) : KalmanAlignmentMetricsUpdator( config )
{
  short int maxDistance = config.getUntrackedParameter< int >( "MaxMetricsDistance", 5 );
  theMetricsCalculator.setMaxDistance( maxDistance );

  std::vector< unsigned int > dummy;
  theFixedAlignableIds = config.getUntrackedParameter< std::vector<unsigned int> >( "FixedAlignableIds", dummy );

  edm::LogInfo("Alignment") << "@SUB=SimpleMetricsUpdator::SimpleMetricsUpdator "
                            << "\nInstance of MetricsCalculator created (MaxMetricsDistance = " << maxDistance << ").";
}

void SimpleMetricsUpdator::update( const std::vector< Alignable* > & alignables )
{
  std::vector< Alignable* > alignablesForUpdate;
  std::vector< Alignable* >::const_iterator it;

  for ( it = alignables.begin(); it != alignables.end(); ++it )
  {
    unsigned int subdetId = static_cast< unsigned int >( (*it)->geomDetId().subdetId() );

    if ( std::find( theFixedAlignableIds.begin(), theFixedAlignableIds.end(), subdetId ) == theFixedAlignableIds.end() )
    {
      alignablesForUpdate.push_back( *it );
    }
  }

  theMetricsCalculator.updateDistances( alignablesForUpdate );
}


const std::vector< Alignable* >
SimpleMetricsUpdator::additionalAlignables( const std::vector< Alignable* > & alignables )
{
  std::vector< Alignable* > result;
  std::vector< Alignable* >::const_iterator itAD;

  std::map< Alignable*, short int > updateList;
  std::map< Alignable*, short int >::iterator itUL;

  std::set< Alignable* > alignablesFromUpdateList;
  std::set< Alignable* >::iterator itAUL;

  // make union of all lists
  for ( itAD = alignables.begin(); itAD != alignables.end(); itAD++ )
  {
    updateList = theMetricsCalculator.getDistances( *itAD );
    for ( itUL = updateList.begin(); itUL != updateList.end(); itUL++ )
    {
      alignablesFromUpdateList.insert( itUL->first );
    }
    updateList.clear();
  }

  // make final list of modules for update
  for ( itAUL = alignablesFromUpdateList.begin(); itAUL != alignablesFromUpdateList.end(); itAUL++ )
  {
    if ( find( alignables.begin(), alignables.end(), *itAUL ) == alignables.end() )
    {
      result.push_back( *itAUL );
    }
  }

  return result;
}


const std::map< Alignable*, short int >
SimpleMetricsUpdator::additionalAlignablesWithDistances( const std::vector< Alignable* > & alignables )
{
  std::map< Alignable*, short int > result;
  std::map< Alignable*, short int > updateList;
  std::map< Alignable*, short int >::iterator itUL;
  std::map< Alignable*, short int >::iterator itFind;

  std::vector< Alignable* >::const_iterator itAD;

  // make union of all lists
  for ( itAD = alignables.begin(); itAD != alignables.end(); itAD++ )
  {
    updateList = theMetricsCalculator.getDistances( *itAD );
    for ( itUL = updateList.begin(); itUL != updateList.end(); itUL++ )
    {
      itFind = result.find( itUL->first );
      if ( itFind == result.end() )
      {
	result[itUL->first] = itUL->second;
      }
      else if ( itFind->second < itUL->second )
      {
	itFind->second = itUL->second;
      }
    }
  }

  for ( itAD = alignables.begin(); itAD != alignables.end(); itAD++ )
  {
    itFind = result.find( *itAD );
    if ( itFind != result.end() ) result.erase( itFind );
  }

  return result;
}
