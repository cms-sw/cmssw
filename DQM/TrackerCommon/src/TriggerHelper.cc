//
// $Id: TriggerHelper.cc,v 1.6 2010/03/08 20:53:57 vadler Exp $
//


#include "DQM/TrackerCommon/interface/TriggerHelper.h"

#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"


using namespace std;
using namespace edm;


/// To be called from the ED module's c'tor
TriggerHelper::TriggerHelper( const ParameterSet & config )
  : on_( true )
  , onDcs_( true )
  , onGt_( true )
  , onL1_( true )
  , onHlt_( true )
{

  if ( config.exists( "andOr" ) ) {
    andOr_ = config.getParameter< bool >( "andOr" );
  } else {
    on_    = false;
    onDcs_ = false;
    onGt_  = false;
    onL1_  = false;
    onHlt_ = false;
  }
  if ( config.exists( "andOrDcs" ) ) {
    andOrDcs_      = config.getParameter< bool >( "andOrDcs" );
    dcsInputTag_   = config.getParameter< InputTag >( "dcsInputTag" );
    dcsPartitions_ = config.getParameter< vector< int > >( "dcsPartitions" );
    errorReplyDcs_ = config.getParameter< bool >( "errorReplyDcs" );
    // An empty DCS partitions list acts as switch.
    if ( dcsPartitions_.empty() ) onDcs_ = false;
  } else {
    onDcs_ = false;
  }
  if ( config.exists( "andOrGt" ) ) {
    andOrGt_              = config.getParameter< bool >( "andOrGt" );
    gtInputTag_           = config.getParameter< InputTag >( "gtInputTag" );
    gtLogicalExpressions_ = config.getParameter< vector< string > >( "gtStatusBits" );
    errorReplyGt_         = config.getParameter< bool >( "errorReplyGt" );
    // An empty GT status bits logical expressions list acts as switch.
    if ( gtLogicalExpressions_.empty() ) onGt_ = false;
  } else {
    onGt_ = false;
  }
  if ( config.exists( "andOrL1" ) ) {
    andOrL1_              = config.getParameter< bool >( "andOrL1" );
    l1LogicalExpressions_ = config.getParameter< vector< string > >( "l1Algorithms" );
    errorReplyL1_         = config.getParameter< bool >( "errorReplyL1" );
    // An empty L1 logical expressions list acts as switch.
    if ( l1LogicalExpressions_.empty() ) onL1_ = false;
  } else {
    onL1_ = false;
  }
  if ( config.exists( "andOrHlt" ) ) {
    andOrHlt_              = config.getParameter< bool >( "andOrHlt" );
    hltInputTag_           = config.getParameter< InputTag >( "hltInputTag" );
    hltLogicalExpressions_ = config.getParameter< vector< string > >( "hltPaths" );
    errorReplyHlt_         = config.getParameter< bool >( "errorReplyHlt" );
    // An empty HLT logical expressions list acts as switch.
    if ( hltLogicalExpressions_.empty() ) onHlt_ = false;
  } else {
    onHlt_ = false;
  }

}


/// To be called from beginRun() methods
void TriggerHelper::initRun( const Run & run, const EventSetup & setup )
{

  hltConfigInit_ = false;
  if ( ! onHlt_ ) return;
  if ( hltInputTag_.process().size() == 0 ) {
    LogError( "hltProcess" ) << "HLT TriggerResults InputTag \"" << hltInputTag_.encode() << "\" specifies no process";
    return;
  }
  bool changed( true );
  if ( ! hltConfig_.init( run, setup, hltInputTag_.process(), changed ) ) {
    LogError( "hltConfigInit" ) << "HLT config initialization error with process name \"" << hltInputTag_.process() << "\"";
    return;
  }
  if ( hltConfig_.size() <= 0 ) {
    LogError( "hltConfigSize" ) << "HLT config size error";
    return;
  }
  hltConfigInit_ = true;

}


/// To be called from analyze/filter() methods
bool TriggerHelper::accept( const Event & event, const EventSetup & setup )
{

  // Getting the and/or switch from the configuration
  // If it does not exist, the configuration is considered not to be present,
  // and the filter dos not have any effect.
  if ( ! on_ ) return true;

  // Determine decision
  if ( andOr_ ) return ( acceptDcs( event ) || acceptGt( event ) || acceptL1( event, setup ) || acceptHlt( event ) );
  return ( acceptDcs( event ) && acceptGt( event ) && acceptL1( event, setup ) && acceptHlt( event ) );

}


bool TriggerHelper::acceptDcs( const Event & event )
{

  // Getting the and/or DCS switch from the configuration
  // If it does not exist, the configuration is considered not to be present,
  // and the filter dos not have any effect
  if ( ! onDcs_ ) return ( ! andOr_ ); // logically neutral, depending on base logical connective

  // Accessing the DcsStatusCollection
  Handle< DcsStatusCollection > dcsStatus;
  event.getByLabel( dcsInputTag_, dcsStatus );
  if ( ! dcsStatus.isValid() ) {
    LogError( "dcsStatusValid" ) << "DcsStatusCollection product with InputTag \"" << dcsInputTag_.encode() << "\" not in event ==> decision: " << errorReplyDcs_;
    return errorReplyDcs_;
  }

  // Determine decision of DCS partition combination and return
  if ( andOrDcs_ ) { // OR combination
    for ( vector< int >::const_iterator partitionNumber = dcsPartitions_.begin(); partitionNumber != dcsPartitions_.end(); ++partitionNumber ) {
      if ( acceptDcsPartition( dcsStatus, *partitionNumber ) ) return true;
    }
    return false;
  }
  for ( vector< int >::const_iterator partitionNumber = dcsPartitions_.begin(); partitionNumber != dcsPartitions_.end(); ++partitionNumber ) {
    if ( ! acceptDcsPartition( dcsStatus, *partitionNumber ) ) return false;
  }
  return true;

}


bool TriggerHelper::acceptDcsPartition( const Handle< DcsStatusCollection > & dcsStatus, int dcsPartition ) const
{

  // Error checks
  switch( dcsPartition ) {
    case DcsStatus::EBp   :
    case DcsStatus::EBm   :
    case DcsStatus::EEp   :
    case DcsStatus::EEm   :
    case DcsStatus::HBHEa :
    case DcsStatus::HBHEb :
    case DcsStatus::HBHEc :
    case DcsStatus::HF    :
    case DcsStatus::HO    :
    case DcsStatus::RPC   :
    case DcsStatus::DT0   :
    case DcsStatus::DTp   :
    case DcsStatus::DTm   :
    case DcsStatus::CSCp  :
    case DcsStatus::CSCm  :
    case DcsStatus::CASTOR:
    case DcsStatus::TIBTID:
    case DcsStatus::TOB   :
    case DcsStatus::TECp  :
    case DcsStatus::TECm  :
    case DcsStatus::BPIX  :
    case DcsStatus::FPIX  :
    case DcsStatus::ESp   :
    case DcsStatus::ESm   :
      break;
    default:
      LogError( "dcsPartition" ) << "DCS partition number \"" << dcsPartition << "\" does not exist ==> decision: " << errorReplyDcs_;
      return errorReplyDcs_;
  }

  // Determine decision
  return dcsStatus->at( 0 ).ready( dcsPartition );

}


/// Does this event fulfill the configured GT status logical expression combination?
bool TriggerHelper::acceptGt( const Event & event )
{

  // Getting the and/or GT status bits switch from the configuration
  // If it does not exist, the configuration is considered not to be present,
  // and the filter dos not have any effect
  if ( ! onGt_ ) return ( ! andOr_ ); // logically neutral, depending on base logical connective

  // Accessing the L1GlobalTriggerReadoutRecord
  Handle< L1GlobalTriggerReadoutRecord > gtReadoutRecord;
  event.getByLabel( gtInputTag_, gtReadoutRecord );
  if ( ! gtReadoutRecord.isValid() ) {
    LogError( "gtReadoutRecordValid" ) << "L1GlobalTriggerReadoutRecord product with InputTag \"" << gtInputTag_.encode() << "\" not in event ==> decision: " << errorReplyGt_;
    return errorReplyGt_;
  }

  // Determine decision of GT status bits logical expression combination and return
  if ( andOrGt_ ) { // OR combination
    for ( vector< string >::const_iterator gtLogicalExpression = gtLogicalExpressions_.begin(); gtLogicalExpression != gtLogicalExpressions_.end(); ++gtLogicalExpression ) {
      if ( acceptGtLogicalExpression( gtReadoutRecord, *gtLogicalExpression ) ) return true;
    }
    return false;
  }
  for ( vector< string >::const_iterator gtLogicalExpression = gtLogicalExpressions_.begin(); gtLogicalExpression != gtLogicalExpressions_.end(); ++gtLogicalExpression ) {
    if ( ! acceptGtLogicalExpression( gtReadoutRecord, *gtLogicalExpression ) ) return false;
  }
  return true;

}


/// Does this event fulfill this particular GT status bits' logical expression?
bool TriggerHelper::acceptGtLogicalExpression( const Handle< L1GlobalTriggerReadoutRecord > & gtReadoutRecord, string gtLogicalExpression )
{

  // Check empty strings
  if ( gtLogicalExpression.empty() ) {
    LogError( "gtLogicalExpression" ) << "Empty logical expression ==> decision: " << errorReplyGt_;
    return errorReplyGt_;
  }

  // Negated paths
  bool negExpr( negate( gtLogicalExpression ) );
  if ( negExpr && gtLogicalExpression.empty() ) {
    LogError( "gtLogicalExpression" ) << "Empty (negated) logical expression ==> decision: " << errorReplyGt_;
    return errorReplyGt_;
  }

  // Parse logical expression and determine GT status bit decision
  L1GtLogicParser gtAlgoLogicParser( gtLogicalExpression );
  // Loop over status bits
  for ( size_t iStatusBit = 0; iStatusBit < gtAlgoLogicParser.operandTokenVector().size(); ++iStatusBit ) {
    const string gtStatusBit( gtAlgoLogicParser.operandTokenVector().at( iStatusBit ).tokenName );
    // Manipulate status bit decision as stored in the parser
    bool decision;
    // Hard-coded status bits!!!
    if ( gtStatusBit == "PhysDecl" || gtStatusBit == "PhysicsDeclared" ) {
      decision = ( gtReadoutRecord->gtFdlWord().physicsDeclared() == 1 );
    } else {
      LogError( "gtStatusBit" ) << "GT status bit \"" << gtStatusBit << "\" is not defined ==> decision: " << errorReplyGt_;
      decision = errorReplyDcs_;
    }
    gtAlgoLogicParser.operandTokenVector().at( iStatusBit ).tokenResult = decision;
  }

  // Determine decision
  const bool gtDecision( gtAlgoLogicParser.expressionResult() );
  return negExpr ? ( ! gtDecision ) : gtDecision;

}


/// Was this event accepted by the configured L1 logical expression combination?
bool TriggerHelper::acceptL1( const Event & event, const EventSetup & setup )
{

  // Getting the and/or L1 switch from the configuration
  // If it does not exist, the configuration is considered not to be present,
  // and the filter dos not have any effect
  if ( ! onL1_ ) return ( ! andOr_ ); // logically neutral, depending on base logical connective

  // Getting the L1 event setup
  l1Gt_.retrieveL1EventSetup( setup ); // FIXME This can possibly go to initRun()

  // Determine decision of L1 logical expression combination and return
  if ( andOrL1_ ) { // OR combination
    for ( vector< string >::const_iterator l1LogicalExpression = l1LogicalExpressions_.begin(); l1LogicalExpression != l1LogicalExpressions_.end(); ++l1LogicalExpression ) {
      if ( acceptL1LogicalExpression( event, *l1LogicalExpression ) ) return true;
    }
    return false;
  }
  for ( vector< string >::const_iterator l1LogicalExpression = l1LogicalExpressions_.begin(); l1LogicalExpression != l1LogicalExpressions_.end(); ++l1LogicalExpression ) {
    if ( ! acceptL1LogicalExpression( event, *l1LogicalExpression ) ) return false;
  }
  return true;

}


/// Was this event accepted by this particular L1 algorithms' logical expression?
bool TriggerHelper::acceptL1LogicalExpression( const Event & event, string l1LogicalExpression )
{

  // Check empty strings
  if ( l1LogicalExpression.empty() ) {
    LogError( "l1LogicalExpression" ) << "Empty logical expression ==> decision: " << errorReplyL1_;
    return errorReplyL1_;
  }

  // Negated logical expression
  bool negExpr( negate( l1LogicalExpression ) );
  if ( negExpr && l1LogicalExpression.empty() ) {
    LogError( "l1LogicalExpression" ) << "Empty (negated) logical expression ==> decision: " << errorReplyL1_;
    return errorReplyL1_;
  }

  // Parse logical expression and determine L1 decision
  L1GtLogicParser l1AlgoLogicParser( l1LogicalExpression );
  // Loop over algorithms
  for ( size_t iAlgorithm = 0; iAlgorithm < l1AlgoLogicParser.operandTokenVector().size(); ++iAlgorithm ) {
    const string l1AlgoName( l1AlgoLogicParser.operandTokenVector().at( iAlgorithm ).tokenName );
    int error( -1 );
    const bool decision( l1Gt_.decision( event, l1AlgoName, error ) );
    // Error checks
    if ( error != 0 ) {
      if ( error == 1 ) LogError( "l1AlgorithmInMenu" ) << "L1 algorithm \"" << l1AlgoName << "\" does not exist in the L1 menu ==> decision: "                                          << errorReplyL1_;
      else              LogError( "l1AlgorithmError" )  << "L1 algorithm \"" << l1AlgoName << "\" received error code " << error << " from L1GtUtils::decisionBeforeMask ==> decision: " << errorReplyL1_;
      l1AlgoLogicParser.operandTokenVector().at( iAlgorithm ).tokenResult = errorReplyL1_;
      continue;
    }
    // Manipulate algo decision as stored in the parser
    l1AlgoLogicParser.operandTokenVector().at( iAlgorithm ).tokenResult = decision;
  }

  // Return decision
  const bool l1Decision( l1AlgoLogicParser.expressionResult() );
  return negExpr ? ( ! l1Decision ) : l1Decision;

}


/// Was this event accepted by the configured HLT logical expression combination?
bool TriggerHelper::acceptHlt( const Event & event )
{

  // Getting the and/or HLT switch from the configuration
  // If it does not exist, the configuration is considered not to be present,
  // and the filter dos not have any effect
  if ( ! onHlt_ ) return ( ! andOr_ ); // logically neutral, depending on base logical connective

  // Checking the HLT configuration,
  if ( ! hltConfigInit_ ) {
    LogError( "hltConfigInit" ) << "HLT config error ==> decision: " << errorReplyHlt_;
    return errorReplyHlt_;
  }

  // Accessing the TriggerResults
  Handle< TriggerResults > hltTriggerResults;
  event.getByLabel( hltInputTag_, hltTriggerResults );
  if ( ! hltTriggerResults.isValid() ) {
    LogError( "triggerResultsValid" ) << "TriggerResults product with InputTag \"" << hltInputTag_.encode() << "\" not in event ==> decision: " << errorReplyHlt_;
    return errorReplyHlt_;
  }

  // Determine decision of HLT logical expression combination and return
  if ( andOrHlt_ ) { // OR combination
    for ( vector< string >::const_iterator hltLogicalExpression = hltLogicalExpressions_.begin(); hltLogicalExpression != hltLogicalExpressions_.end(); ++hltLogicalExpression ) {
      if ( acceptHltLogicalExpression( hltTriggerResults, *hltLogicalExpression ) ) return true;
    }
    return false;
  }
  for ( vector< string >::const_iterator hltLogicalExpression = hltLogicalExpressions_.begin(); hltLogicalExpression != hltLogicalExpressions_.end(); ++hltLogicalExpression ) {
    if ( ! acceptHltLogicalExpression( hltTriggerResults, *hltLogicalExpression ) ) return false;
  }
  return true;

}


/// Was this event accepted by this particular HLT paths' logical expression?
bool TriggerHelper::acceptHltLogicalExpression( const Handle< TriggerResults > & hltTriggerResults, string hltLogicalExpression ) const
{

  // Check empty strings
  if ( hltLogicalExpression.empty() ) {
    LogError( "hltLogicalExpression" ) << "Empty logical expression ==> decision: " << errorReplyHlt_;
    return errorReplyHlt_;
  }

  // Negated paths
  bool negExpr( negate( hltLogicalExpression ) );
  if ( negExpr && hltLogicalExpression.empty() ) {
    LogError( "hltLogicalExpression" ) << "Empty (negated) logical expression ==> decision: " << errorReplyHlt_;
    return errorReplyHlt_;
  }

  // Parse logical expression and determine HLT decision
  L1GtLogicParser hltAlgoLogicParser( hltLogicalExpression );
  // Loop over paths
  for ( size_t iPath = 0; iPath < hltAlgoLogicParser.operandTokenVector().size(); ++iPath ) {
    const string hltPathName( hltAlgoLogicParser.operandTokenVector().at( iPath ).tokenName );
    const unsigned indexPath( hltConfig_.triggerIndex( hltPathName ) );
    // Further error checks
    if ( indexPath == hltConfig_.size() ) {
      LogError( "hltPathInProcess" ) << "HLT path \"" << hltPathName << "\" is not found in process " << hltInputTag_.process() << " ==> decision: " << errorReplyHlt_;
      hltAlgoLogicParser.operandTokenVector().at( iPath ).tokenResult = errorReplyHlt_;
      continue;
    }
    if ( hltTriggerResults->error( indexPath ) ) {
      LogError( "hltPathError" ) << "HLT path \"" << hltPathName << "\" in error ==> decision: " << errorReplyHlt_;
      hltAlgoLogicParser.operandTokenVector().at( iPath ).tokenResult = errorReplyHlt_;
      continue;
    }
    // Manipulate algo decision as stored in the parser
    const bool decision( hltTriggerResults->accept( indexPath ) );
    hltAlgoLogicParser.operandTokenVector().at( iPath ).tokenResult = decision;
  }

  // Determine decision
  const bool hltDecision( hltAlgoLogicParser.expressionResult() );
  return negExpr ? ( ! hltDecision ) : hltDecision;

}


/// Checks for negated words
bool TriggerHelper::negate( string & word ) const
{

  bool negate( false );
  if ( word.at( 0 ) == '~' ) {
    negate = true;
    word.erase( 0, 1 );
  }
  return negate;

}
