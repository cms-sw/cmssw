//
// $Id$
//


#include "DQM/TrackerCommon/interface/TriggerHelper.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream> // DEBUG


using namespace std;
using namespace edm;


/// Was this event accepted by the configured HLT paths combination?
bool TriggerHelper::accept( const Event & event, const ParameterSet & config )
{

  // Configuration parameter tags
  const string hltInputTagConfig( "hltInputTag" );
  const string hltPathConfig( "hltPaths" );
  const string andOrConfig( "andOr" );
  const string errorReplyConfig( "errorReply" );

  // Getting the TriggerResults InputTag from the configuration
  // If an according InputTag is found in the configuration, it is expected to be meaningful and correct.
  // If not, the filter is switched off.
  if ( ! config.exists( hltInputTagConfig ) ) return true;
  hltInputTag_ = config.getParameter< InputTag >( hltInputTagConfig );

  // Getting the HLT path name of question from the configuration
  // An empty configuration parameter acts also as switch.
  hltPathNames_ = config.getParameter< vector< string > >( hltPathConfig );
  if ( hltPathNames_.empty() ) return true;

  // Getting remaining configuration parameters
  errorReply_ = config.getParameter< bool >( errorReplyConfig );
  andOr_ = config.getParameter< bool >( andOrConfig );

  // Checking the TriggerResults InputTag
  // The process name has to be given.
  if ( hltInputTag_.process().size() == 0 ) {
    LogError( "hltProcess" ) << "HLT TriggerResults InputTag " << hltInputTag_.encode() << " specifies no process";
    return errorReply_;
  }

  // Accessing the TriggerResults
  event.getByLabel( hltInputTag_, hltTriggerResults_ );
  if ( ! hltTriggerResults_.isValid() ) {
    LogError( "triggerResultsValid" ) << "TriggerResults product with InputTag " << hltInputTag_.encode() << " not in event";
    return errorReply_;
  }

  // Getting the HLT configuration from the provenance
  bool changed( true );
  if ( ! hltConfig_.init( event, hltInputTag_.process(), changed ) ) {
    LogError( "hltConfigInit" ) << "HLT config initialization error with process name " << hltInputTag_.process();
    return errorReply_;
  }
  if ( hltConfig_.size() <= 0 ) {
    LogError( "hltConfigSize" ) << "HLT config size error";
    return errorReply_;
  }

  // Determine acceptance of HLT path combination and return
  if ( andOr_ ) { // OR combination
    for ( vector< string >::const_iterator pathName = hltPathNames_.begin(); pathName != hltPathNames_.end(); ++pathName ) {
      if ( acceptPath( *pathName ) ) return true;
    }
    return false;
  }
  for ( vector< string >::const_iterator pathName = hltPathNames_.begin(); pathName != hltPathNames_.end(); ++pathName ) {
    if ( ! acceptPath( *pathName ) ) return false;
  }
  return true;

}

/// Was this event accepted by this particular HLT path?
bool TriggerHelper::acceptPath( string hltPathName ) const
{

  // Check empty strings
  if ( hltPathName.empty() ) {
    LogError( "hltPathName" ) << "Empty path name";
    return errorReply_;
  }

  // Negated paths
  bool notPath( false );
  if ( hltPathName.at( 0 ) == '~' ) {
    notPath = true;
    hltPathName.erase( 0, 1 );
    // Check empty string again
    if ( hltPathName.empty() ) {
      LogError( "hltPathName" ) << "Empty (negated) path name";
      return errorReply_;
    }
  }

  // Further error checks
  const unsigned indexPath( hltConfig_.triggerIndex( hltPathName ) );
  if ( indexPath == hltConfig_.size() ) {
    LogError( "hltPathInProcess" ) << "Path " << hltPathName << " is not found in process " << hltInputTag_.process();
    return errorReply_;
  }
  if ( hltTriggerResults_->error( indexPath ) ) {
    LogError( "hltPathError" ) << "Path " << hltPathName << " in error";
    return errorReply_;
  }

  // Determine decision
  cout << "TriggerHelper: path "; // DEBUG
  if ( notPath ) cout << "~" << hltPathName; // DEBUG
  else           cout        << hltPathName; // DEBUG
  cout << "->" << hltTriggerResults_->accept( indexPath ) << endl; // DEBUG
  return notPath ? ( ! hltTriggerResults_->accept( indexPath ) ) : hltTriggerResults_->accept( indexPath );

}
