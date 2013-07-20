//
// See header file for description
// $Id: PrescaleWeightProvider.cc,v 1.2 2010/08/25 16:20:29 vadler Exp $
//


#include "CommonTools/TriggerUtils/interface/PrescaleWeightProvider.h"

#include <sstream>

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


PrescaleWeightProvider::PrescaleWeightProvider( const edm::ParameterSet & config )
// default values
: verbosity_( 0 )
, triggerResults_( "TriggerResults::HLT" )
, l1GtTriggerMenuLite_( "l1GtTriggerMenuLite" )
{

  hltPaths_.clear();
  if ( config.exists( "prescaleWeightVerbosityLevel" ) )      verbosity_           = config.getParameter< unsigned >( "prescaleWeightVerbosityLevel" );
  if ( config.exists( "prescaleWeightTriggerResults" ) )      triggerResults_      = config.getParameter< edm::InputTag >( "prescaleWeightTriggerResults" );
  if ( config.exists( "prescaleWeightL1GtTriggerMenuLite" ) ) l1GtTriggerMenuLite_ = config.getParameter< edm::InputTag >( "prescaleWeightL1GtTriggerMenuLite" );
  if ( config.exists( "prescaleWeightHltPaths" ) )            hltPaths_            = config.getParameter< std::vector< std::string > >( "prescaleWeightHltPaths" );

  configured_ = true;
  if ( triggerResults_.process().empty() ) {
    configured_ = false;
    if ( verbosity_ > 0 ) edm::LogWarning( "PrescaleWeightProvider" ) << "Process name not configured via TriggerResults InputTag";
  } else if ( triggerResults_.label().empty() ) {
    configured_ = false;
    if ( verbosity_ > 0 ) edm::LogWarning( "PrescaleWeightProvider" ) << "TriggerResults label not configured";
  } else if ( l1GtTriggerMenuLite_.label().empty() ) {
    configured_ = false;
    if ( verbosity_ > 0 ) edm::LogWarning( "PrescaleWeightProvider" ) << "L1GtTriggerMenuLite label not configured";
  } else if ( hltPaths_.empty() ) {
    configured_ = false;
    if ( verbosity_ > 0 ) edm::LogError( "PrescaleWeightProvider" ) << "HLT paths of interest not configured";
  }

}

void PrescaleWeightProvider::initRun( const edm::Run & run, const edm::EventSetup & setup )
{

  init_ = true;

  if ( ! configured_ ) {
    init_ = false;
    if ( verbosity_ > 0 ) edm::LogWarning( "PrescaleWeightProvider" ) << "Run initialisation failed due to failing configuration";
    return;
  }

  bool hltChanged( false );
  if ( ! hltConfig_.init( run, setup, triggerResults_.process(), hltChanged ) ) {
    if ( verbosity_ > 0 ) edm::LogError( "PrescaleWeightProvider" ) << "HLT config initialization error with process name \"" << triggerResults_.process() << "\"";
    init_ = false;
  } else if ( hltConfig_.size() <= 0 ) {
    if ( verbosity_ > 0 ) edm::LogError( "PrescaleWeightProvider" ) << "HLT config size error";
    init_ = false;
  } else if ( hltChanged ) {
    if ( verbosity_ > 0 ) edm::LogInfo( "PrescaleWeightProvider" ) << "HLT configuration changed";
  }
  if ( ! init_ ) return;

  run.getByLabel( l1GtTriggerMenuLite_.label(), triggerMenuLite_ );
  if ( ! triggerMenuLite_.isValid() ) {
    if ( verbosity_ > 0 ) edm::LogError( "PrescaleWeightProvider" ) << "L1GtTriggerMenuLite with label \"" << l1GtTriggerMenuLite_.label() << "\" not found";
    init_ = false;
  }

}


int PrescaleWeightProvider::prescaleWeight( const edm::Event & event, const edm::EventSetup & setup )
{
  if ( ! init_ ) return 1;

  // L1
  L1GtUtils l1GtUtils;
  l1GtUtils.retrieveL1EventSetup( setup );

  // HLT
  edm::Handle< edm::TriggerResults > triggerResults;
  event.getByLabel( triggerResults_, triggerResults);
  if( ! triggerResults.isValid() ) {
    if ( verbosity_ > 0 ) edm::LogError("PrescaleWeightProvider::prescaleWeight") << "TriggerResults product not found for InputTag \"" << triggerResults_.encode() << "\"";
    return 1;
  }

  const int SENTINEL( -1 );
  int weight( SENTINEL );

  for ( unsigned ui = 0; ui < hltPaths_.size(); ui++ ) {
    const std::string hltPath( hltPaths_.at( ui ) );
    unsigned hltIndex( hltConfig_.triggerIndex( hltPath ) );
    if ( hltIndex == hltConfig_.size() ) {
      if ( verbosity_ > 0 ) edm::LogError( "PrescaleWeightProvider::prescaleWeight" ) << "HLT path \"" << hltPath << "\" does not exist";
      continue;
    }
    if ( ! triggerResults->accept( hltIndex ) ) continue;

    const std::vector< std::pair < bool, std::string > > level1Seeds = hltConfig_.hltL1GTSeeds( hltPath );
    if ( level1Seeds.size() != 1 ) {
      if ( verbosity_ > 0 ) edm::LogError( "PrescaleWeightProvider::prescaleWeight" ) << "HLT path \"" << hltPath << "\" provides too many L1 seeds";
      return 1;
    }
    parseL1Seeds( level1Seeds.at( 0 ).second );
    if ( l1SeedPaths_.empty() ){
      if ( verbosity_ > 0 ) edm::LogWarning("PrescaleWeightProvider::prescaleWeight") << "Failed to parse L1 seeds for HLT path \"" << hltPath << "\"";
      continue;
    }

    int l1Prescale( SENTINEL );
    for ( unsigned uj = 0; uj < l1SeedPaths_.size(); uj++ ) {
      int l1TempPrescale( SENTINEL );
      int errorCode( 0 );
      if ( level1Seeds.at( 0 ).first ) { // technical triggers
        unsigned          techBit( atoi( l1SeedPaths_.at( uj ).c_str() ) );
        const std::string techName( *( triggerMenuLite_->gtTechTrigName( techBit, errorCode ) ) );
        if ( errorCode != 0 ) continue;
	if ( ! l1GtUtils.decision( event, techName, errorCode ) ) continue;
	if ( errorCode != 0 ) continue;
	l1TempPrescale = l1GtUtils.prescaleFactor( event, techName, errorCode );
	if ( errorCode != 0 ) continue;
      }
      else { // algorithmic triggers
        if ( ! l1GtUtils.decision( event, l1SeedPaths_.at( uj ), errorCode ) ) continue;
        if ( errorCode != 0 ) continue;
	l1TempPrescale = l1GtUtils.prescaleFactor( event, l1SeedPaths_.at( uj ), errorCode );
	if ( errorCode != 0 ) continue;
      }
      if ( l1TempPrescale > 0 ){
        if ( l1Prescale == SENTINEL || l1Prescale > l1TempPrescale ) l1Prescale = l1TempPrescale;
      }
    }
    if ( l1Prescale == SENTINEL ){
      if ( verbosity_ > 0 ) edm::LogError( "PrescaleWeightProvider::prescaleWeight" ) << "Unable to find the L1 prescale for HLT path \"" << hltPath << "\"";
      continue;
    }

    int hltPrescale( hltConfig_.prescaleValue( event, setup, hltPath ) );

    if ( hltPrescale * l1Prescale > 0 ) {
      if ( weight == SENTINEL || weight > hltPrescale * l1Prescale ) {
        weight = hltPrescale * l1Prescale;
      }
    }
  }

  if ( weight == SENTINEL ){
    if ( verbosity_ > 0 ) edm::LogWarning( "PrescaleWeightProvider::prescaleWeight" ) << "No valid weight for any requested HLT path, returning default weight of 1";
    return 1;
  }
  return weight;

}


void PrescaleWeightProvider::parseL1Seeds( const std::string & l1Seeds )
{
  l1SeedPaths_.clear();
  std::stringstream ss( l1Seeds );
  std::string       buf;

  while ( ss.good() && ! ss.eof() ){
    ss >> buf;
    if ( buf[0] == '('  || buf[ buf.size() - 1 ] == ')' || buf == "AND" || buf == "NOT" ){
      l1SeedPaths_.clear();
      if ( verbosity_ > 0 ) edm::LogWarning( "PrescaleWeightProvider::parseL1Seeds" ) << "Only supported logical expression is OR";
      return;
    }
    else if (buf == "OR") continue;
    else                  l1SeedPaths_.push_back( buf );
  }

}
