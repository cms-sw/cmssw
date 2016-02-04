//#include "Alignment/KalmanAlignmentAlgorithm/plugins/MultiMetricsUpdator.h"
#include "MultiMetricsUpdator.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdatorPlugin.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"



MultiMetricsUpdator::MultiMetricsUpdator( const edm::ParameterSet & config ) : KalmanAlignmentMetricsUpdator( config )
{
  std::vector<std::string> strConfig = config.getParameter< std::vector<std::string> >( "Configurations" );
  std::vector<std::string>::iterator itConfig;
  for ( itConfig = strConfig.begin(); itConfig != strConfig.end(); ++itConfig )
  {
    edm::ParameterSet updatorConfig = config.getParameter<edm::ParameterSet>( *itConfig );
    theMetricsUpdators.push_back( new SimpleMetricsUpdator( updatorConfig ) );
  }


  edm::LogInfo("Alignment") << "@SUB=MultiMetricsUpdator::MultiMetricsUpdator "
                            << "\nInstance of MultiMetricsUpdator created.";;
}


MultiMetricsUpdator::~MultiMetricsUpdator( void )
{
  std::vector< SimpleMetricsUpdator* >::const_iterator it;

  for ( it = theMetricsUpdators.begin(); it != theMetricsUpdators.end(); ++it )
    delete *it;
}


void MultiMetricsUpdator::update( const std::vector< Alignable* > & alignables )
{
  std::vector< SimpleMetricsUpdator* >::const_iterator it;
  for ( it = theMetricsUpdators.begin(); it != theMetricsUpdators.end(); ++it )
  {
    (*it)->update( alignables );
  }
}


const std::vector< Alignable* >
MultiMetricsUpdator::additionalAlignables( const std::vector< Alignable* > & alignables )
{
  std::set< Alignable* > alignableSet;

  std::vector< SimpleMetricsUpdator* >::const_iterator it;
  for ( it = theMetricsUpdators.begin(); it != theMetricsUpdators.end(); ++it )
  {
    const std::vector< Alignable* > additional = (*it)->additionalAlignables( alignables );
    alignableSet.insert( additional.begin(), additional.end() );
  }

  std::vector< Alignable* > result;
  result.insert( result.end(), alignableSet.begin(), alignableSet.end() );
  return result;
}


const std::vector< Alignable* >
MultiMetricsUpdator::alignables( void ) const
{
  std::set< Alignable* > alignableSet;

  std::vector< SimpleMetricsUpdator* >::const_iterator it;
  for ( it = theMetricsUpdators.begin(); it != theMetricsUpdators.end(); ++it )
  {
    const std::vector< Alignable* > alignables = (*it)->alignables();
    alignableSet.insert( alignables.begin(), alignables.end() );
  }

  std::vector< Alignable* > result;
  result.insert( result.end(), alignableSet.begin(), alignableSet.end() );
  return result;
}


DEFINE_EDM_PLUGIN( KalmanAlignmentMetricsUpdatorPlugin, MultiMetricsUpdator, "MultiMetricsUpdator" );
