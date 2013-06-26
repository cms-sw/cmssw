//#include "Alignment/KalmanAlignmentAlgorithm/plugins/SimpleMetricsUpdator.h"
#include "SimpleMetricsUpdator.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdatorPlugin.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"



SimpleMetricsUpdator::SimpleMetricsUpdator( const edm::ParameterSet & config ) : 
  KalmanAlignmentMetricsUpdator( config ),
  theMinDeltaPerp(0.), theMaxDeltaPerp(0.), theMinDeltaZ(0.), theMaxDeltaZ(0.),
  theGeomDist(0.), theMetricalThreshold(0)
{
  short int maxDistance = config.getUntrackedParameter< int >( "MaxMetricsDistance", 3 );
  theMetricsCalculator.setMaxDistance( maxDistance );

  std::vector< unsigned int > dummy;
  theExcludedSubdetIds = config.getUntrackedParameter< std::vector<unsigned int> >( "ExcludedSubdetIds", dummy );

  theASCFlag =  config.getUntrackedParameter< bool >( "ApplyAdditionalSelectionCriterion", false );
  if ( theASCFlag )
  {
    theMinDeltaPerp = config.getParameter< double >( "MinDeltaPerp" );
    theMaxDeltaPerp = config.getParameter< double >( "MaxDeltaPerp" );
    theMinDeltaZ = config.getParameter< double >( "MinDeltaZ" );
    theMaxDeltaZ = config.getParameter< double >( "MaxDeltaZ" );
    theGeomDist = config.getParameter< double >( "GeomDist" );
    theMetricalThreshold = config.getParameter< unsigned int >( "MetricalThreshold" );
  }

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

    if ( std::find( theExcludedSubdetIds.begin(), theExcludedSubdetIds.end(), subdetId ) == theExcludedSubdetIds.end() )
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
      // extra selection criterion
      if ( theASCFlag && !additionalSelectionCriterion( *itAD, itUL->first, itUL->second ) ) continue;

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


bool
SimpleMetricsUpdator::additionalSelectionCriterion( Alignable* const& referenceAli,
						    Alignable* const& additionalAli,
						    short int metricalDist ) const
{
  if ( metricalDist <= theMetricalThreshold ) return true;

  const DetId detId( referenceAli->geomDetId() );

  const align::PositionType& pos1 = referenceAli->globalPosition(); 
  const align::PositionType& pos2 = additionalAli->globalPosition(); 

  bool barrelRegion = ( detId.subdetId()%2 != 0 );

  if ( barrelRegion )
  {
    double perp1 = pos1.perp();
    double perp2 = pos2.perp();
    double deltaPerp = perp2 - perp1;

    if ( ( deltaPerp < theMinDeltaPerp ) || ( deltaPerp > theMaxDeltaPerp ) ) return false;
  }
  else
  {
    double z1 = pos1.z();
    double z2 = pos2.z();
    double signZ = ( z1 > 0. ) ? 1. : -1;
    double deltaZ = signZ*( z2 - z1 );

    if ( ( deltaZ < theMinDeltaZ ) || ( deltaZ > theMaxDeltaZ ) ) return false;
  }

  double r1 = pos1.mag();
  double r2 = pos2.mag();
  double sp = pos1.x()*pos2.x() + pos1.y()*pos2.y() + pos1.z()*pos2.z();

  double dist = sqrt( r2*r2 - sp*sp/r1/r1 );
  return ( dist < theGeomDist );
}


DEFINE_EDM_PLUGIN( KalmanAlignmentMetricsUpdatorPlugin, SimpleMetricsUpdator, "SimpleMetricsUpdator" );
