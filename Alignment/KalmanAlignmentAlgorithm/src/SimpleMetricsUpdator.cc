#include "Alignment/KalmanAlignmentAlgorithm/interface/SimpleMetricsUpdator.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <set>


SimpleMetricsUpdator::SimpleMetricsUpdator( const edm::ParameterSet & config ) : KalmanAlignmentMetricsUpdator( config )
{
  short int maxDistance = config.getUntrackedParameter< int >( "MaxMetricsDistance", 5 );
  theMetricsCalculator.setMaxDistance( maxDistance );

  std::vector< unsigned int > dummy;
  theFixedAlignableDetIds = config.getUntrackedParameter< std::vector<unsigned int> >( "FixedAlignableDetIds", dummy );

  edm::LogInfo("Alignment") << "@SUB=SimpleMetricsUpdator::SimpleMetricsUpdator "
                            << "\nInstance of MetricsCalculator created (MaxMetricsDistance = " << maxDistance << ").";
}

void SimpleMetricsUpdator::update( const std::vector< AlignableDet* > & alignableDets )
{
  std::vector< AlignableDet* > alignableDetsForUpdate;
  std::vector< AlignableDet* >::const_iterator it;

  for ( it = alignableDets.begin(); it != alignableDets.end(); ++it )
  {
    unsigned int subdetId = static_cast< unsigned int >( (*it)->geomDetId().subdetId() );

    if ( std::find( theFixedAlignableDetIds.begin(), theFixedAlignableDetIds.end(), subdetId ) == theFixedAlignableDetIds.end() )
    {
      alignableDetsForUpdate.push_back( *it );
    }
  }

  theMetricsCalculator.updateDistances( alignableDetsForUpdate );
}


const std::vector< AlignableDet* >
SimpleMetricsUpdator::additionalAlignableDets( const std::vector< AlignableDet* > & alignableDets )
{
  std::vector< AlignableDet* > result;
  std::vector< AlignableDet* >::const_iterator itAD;

  std::map< AlignableDet*, short int > updateList;
  std::map< AlignableDet*, short int >::iterator itUL;

  std::set< AlignableDet* > alignableDetsFromUpdateList;
  std::set< AlignableDet* >::iterator itAUL;

  // make union of all lists
  for ( itAD = alignableDets.begin(); itAD != alignableDets.end(); itAD++ )
  {
    updateList = theMetricsCalculator.getDistances( *itAD );
    for ( itUL = updateList.begin(); itUL != updateList.end(); itUL++ )
    {
      alignableDetsFromUpdateList.insert( itUL->first );
    }
    updateList.clear();
  }

  // make final list of modules for update
  for ( itAUL = alignableDetsFromUpdateList.begin(); itAUL != alignableDetsFromUpdateList.end(); itAUL++ )
  {
    if ( find( alignableDets.begin(), alignableDets.end(), *itAUL ) == alignableDets.end() )
    {
      result.push_back( *itAUL );
    }
  }

  return result;
}


const std::map< AlignableDet*, short int >
SimpleMetricsUpdator::additionalAlignableDetsWithDistances( const std::vector< AlignableDet* > & alignableDets )
{
  std::map< AlignableDet*, short int > result;
  std::map< AlignableDet*, short int > updateList;
  std::map< AlignableDet*, short int >::iterator itUL;
  std::map< AlignableDet*, short int >::iterator itFind;

  std::vector< AlignableDet* >::const_iterator itAD;

  // make union of all lists
  for ( itAD = alignableDets.begin(); itAD != alignableDets.end(); itAD++ )
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

  for ( itAD = alignableDets.begin(); itAD != alignableDets.end(); itAD++ )
  {
    itFind = result.find( *itAD );
    if ( itFind != result.end() ) result.erase( itFind );
  }

  return result;
}
