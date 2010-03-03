//
// $Id: TriggerHelper.cc,v 1.10 2010/02/16 20:17:14 vadler Exp $
//


#include "DQM/TrackerCommon/interface/TriggerHelper.h"

#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"


using namespace std;
using namespace edm;


TriggerHelper::TriggerHelper()
{

  gtReadoutRecord_.clear();
  hltTriggerResults_.clear();
  dcsStatus_.clear();

}


/// DCS, status bits, L1 and HLT filters combined
bool TriggerHelper::accept( const edm::Event & event, const edm::EventSetup & setup, const edm::ParameterSet & config, const HLTConfigProvider & hltConfig, bool hltConfigInit )
{

  // Getting the and/or switch from the configuration
  // If it does not exist, the configuration is considered not to be present,
  // and the filter dos not have any effect.
  if ( ! config.exists( "andOr" ) ) return true;
  andOr_ = config.getParameter< bool >( "andOr" );

  // Determine decision
  if ( andOr_ ) return ( acceptDcs( event, config ) || acceptGt( event, config ) || acceptL1( event, setup, config ) || acceptHlt( event, config, hltConfig, hltConfigInit ) );
  return ( acceptDcs( event, config ) && acceptGt( event, config ) && acceptL1( event, setup, config ) && acceptHlt( event, config, hltConfig, hltConfigInit ) );

}


/// DCS, status bits and HLT filters only
bool TriggerHelper::accept( const edm::Event & event, const edm::ParameterSet & config, const HLTConfigProvider & hltConfig, bool hltConfigInit )
{

  // Getting the and/or switch from the configuration
  // If it does not exist, the configuration is considered not to be present,
  // and the filter dos not have any effect.
  if ( ! config.exists( "andOr" ) ) return true;
  andOr_ = config.getParameter< bool >( "andOr" );

  // Determine decision
  if ( andOr_ ) return ( acceptDcs( event, config ) || acceptGt( event, config ) || acceptHlt( event, config, hltConfig, hltConfigInit ) );
  return ( acceptDcs( event, config ) && acceptGt( event, config ) && acceptHlt( event, config, hltConfig, hltConfigInit ) );

}



/// DCS, GT status and L1 filters combined
bool TriggerHelper::accept( const Event & event, const EventSetup & setup, const ParameterSet & config )
{

  // Getting the and/or switch from the configuration
  // If it does not exist, the configuration is considered not to be present,
  // and the filter dos not have any effect.
  if ( ! config.exists( "andOr" ) ) return true;
  andOr_ = config.getParameter< bool >( "andOr" );

  // Determine decision
  if ( andOr_ ) return ( acceptDcs( event, config ) || acceptGt( event, config ) || acceptL1( event, setup, config ) );
  return ( acceptDcs( event, config ) && acceptGt( event, config ) && acceptL1( event, setup, config ) );

}


/// DCS and GT status filters only
bool TriggerHelper::accept( const Event & event, const ParameterSet & config )
{

  // Getting the and/or switch from the configuration
  // If it does not exist, the configuration is considered not to be present,
  // and the filter dos not have any effect.
  if ( ! config.exists( "andOr" ) ) return true;
  andOr_ = config.getParameter< bool >( "andOr" );

  // Determine decision
  if ( andOr_ ) return ( acceptDcs( event, config ) || acceptGt( event, config ) );
  return ( acceptDcs( event, config ) && acceptGt( event, config ) );

}


bool TriggerHelper::acceptDcs( const edm::Event & event, const edm::ParameterSet & config )
{

  // Getting the and/or DCS switch from the configuration
  // If it does not exist, the configuration is considered not to be present,
  // and the filter dos not have any effect
  if ( ! config.exists( "andOrDcs" ) ) return ( ! andOr_ ); // logically neutral, depending on base logical connective

  // Getting configuration parameters
  const InputTag dcsInputTag( config.getParameter< InputTag >( "dcsInputTag" ) );
  const vector< int > dcsPartitions( config.getParameter< vector< int > >( "dcsPartitions" ) );
  errorReplyDcs_ = config.getParameter< bool >( "errorReplyDcs" );

  // An empty DCS partitions list acts as switch.
  if ( dcsPartitions.empty() ) return ( ! andOr_ );

  // Accessing the DcsStatusCollection
  event.getByLabel( dcsInputTag, dcsStatus_ );
  if ( ! dcsStatus_.isValid() ) {
    LogError( "dcsStatusValid" ) << "DcsStatusCollection product with InputTag " << dcsInputTag.encode() << " not in event ==> decision: " << errorReplyDcs_;
    return errorReplyDcs_;
  }

  // Determine decision of DCS partition combination and return
  if ( config.getParameter< bool >( "andOrDcs" ) ) { // OR combination
    for ( vector< int >::const_iterator partitionNumber = dcsPartitions.begin(); partitionNumber != dcsPartitions.end(); ++partitionNumber ) {
      if ( acceptDcsPartition( *partitionNumber ) ) return true;
    }
    return false;
  }
  for ( vector< int >::const_iterator partitionNumber = dcsPartitions.begin(); partitionNumber != dcsPartitions.end(); ++partitionNumber ) {
    if ( ! acceptDcsPartition( *partitionNumber ) ) return false;
  }
  return true;

}


bool TriggerHelper::acceptDcsPartition( int dcsPartition ) const
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
      LogError( "dcsPartition" ) << "DCS partition number " << dcsPartition << " does not exist ==> decision: " << errorReplyDcs_;
      return errorReplyDcs_;
  }

  // Determine decision
  return dcsStatus_->at( 0 ).ready( dcsPartition );

}


/// Does this event fulfill the configured GT status logical expression combination?
bool TriggerHelper::acceptGt( const Event & event, const ParameterSet & config )
{

  // Getting the and/or GT status bits switch from the configuration
  // If it does not exist, the configuration is considered not to be present,
  // and the filter dos not have any effect
  if ( ! config.exists( "andOrGt" ) ) return ( ! andOr_ ); // logically neutral, depending on base logical connective

  // Getting configuration parameters
  const InputTag gtInputTag( config.getParameter< InputTag >( "gtInputTag" ) );
  const vector< string > gtLogicalExpressions( config.getParameter< vector< string > >( "gtStatusBits" ) );
  errorReplyGt_ = config.getParameter< bool >( "errorReplyGt" );

  // An empty GT status bits logical expressions list acts as switch.
  if ( gtLogicalExpressions.empty() ) return ( ! andOr_ );

  // Accessing the L1GlobalTriggerReadoutRecord
  event.getByLabel( gtInputTag, gtReadoutRecord_ );
  if ( ! gtReadoutRecord_.isValid() ) {
    LogError( "gtReadoutRecordValid" ) << "L1GlobalTriggerReadoutRecord product with InputTag " << gtInputTag.encode() << " not in event ==> decision: " << errorReplyGt_;
    return errorReplyGt_;
  }

  // Determine decision of GT status bits logical expression combination and return
  if ( config.getParameter< bool >( "andOrGt" ) ) { // OR combination
    for ( vector< string >::const_iterator gtLogicalExpression = gtLogicalExpressions.begin(); gtLogicalExpression != gtLogicalExpressions.end(); ++gtLogicalExpression ) {
      if ( acceptGtLogicalExpression( *gtLogicalExpression ) ) return true;
    }
    return false;
  }
  for ( vector< string >::const_iterator gtLogicalExpression = gtLogicalExpressions.begin(); gtLogicalExpression != gtLogicalExpressions.end(); ++gtLogicalExpression ) {
    if ( ! acceptGtLogicalExpression( *gtLogicalExpression ) ) return false;
  }
  return true;

}


/// Does this event fulfill this particular GT status bits' logical expression?
bool TriggerHelper::acceptGtLogicalExpression( string gtLogicalExpression )
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
  // Loop over paths
  for ( size_t iStatusBit = 0; iStatusBit < gtAlgoLogicParser.operandTokenVector().size(); ++iStatusBit ) {
    const string gtStatusBit( gtAlgoLogicParser.operandTokenVector().at( iStatusBit ).tokenName );
    // Manipulate status bit decision as stored in the parser
    bool decision;
    if ( gtStatusBit == "PhysDecl" || gtStatusBit == "PhysicsDeclared" ) {
      decision = ( gtReadoutRecord_->gtFdlWord().physicsDeclared() == 1 );
    } else {
      LogError( "gtStatusBit" ) << "GT status bit " << gtStatusBit << " is not defined ==> decision: " << errorReplyGt_;
      decision = errorReplyDcs_;
    }
    gtAlgoLogicParser.operandTokenVector().at( iStatusBit ).tokenResult = decision;
  }

  // Determine decision
  const bool gtDecision( gtAlgoLogicParser.expressionResult() );
  return negExpr ? ( ! gtDecision ) : gtDecision;

}


/// Was this event accepted by the configured L1 logical expression combination?
bool TriggerHelper::acceptL1( const Event & event, const EventSetup & setup, const ParameterSet & config )
{

  // Getting the and/or L1 switch from the configuration
  // If it does not exist, the configuration is considered not to be present,
  // and the filter dos not have any effect
  if ( ! config.exists( "andOrL1" ) ) return ( ! andOr_ ); // logically neutral, depending on base logical connective

  // Getting configuration parameters
  const vector< string > l1LogicalExpressions( config.getParameter< vector< string > >( "l1Algorithms" ) );
  errorReplyL1_ = config.getParameter< bool >( "errorReplyL1" );

  // An empty L1 logical expressions list acts as switch.
  if ( l1LogicalExpressions.empty() ) return ( ! andOr_ );

  // Getting the L1 event setup
  l1Gt_.retrieveL1EventSetup( setup );

  // Determine decision of L1 logical expression combination and return
  if ( config.getParameter< bool >( "andOrL1" ) ) { // OR combination
    for ( vector< string >::const_iterator l1LogicalExpression = l1LogicalExpressions.begin(); l1LogicalExpression != l1LogicalExpressions.end(); ++l1LogicalExpression ) {
      if ( acceptL1LogicalExpression( event, *l1LogicalExpression ) ) return true;
    }
    return false;
  }
  for ( vector< string >::const_iterator l1LogicalExpression = l1LogicalExpressions.begin(); l1LogicalExpression != l1LogicalExpressions.end(); ++l1LogicalExpression ) {
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
      if ( error == 1 ) LogError( "l1AlgorithmInMenu" ) << "L1 algorithm " << l1AlgoName << " does not exist in the L1 menu ==> decision: "                                          << errorReplyL1_;
      else              LogError( "l1AlgorithmError" )  << "L1 algorithm " << l1AlgoName << " received error code " << error << " from L1GtUtils::decisionBeforeMask ==> decision: " << errorReplyL1_;
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
bool TriggerHelper::acceptHlt( const Event & event, const ParameterSet & config, const HLTConfigProvider & hltConfig, bool hltConfigInit )
{

  // Getting the and/or HLT switch from the configuration
  // If it does not exist, the configuration is considered not to be present,
  // and the filter dos not have any effect
  if ( ! config.exists( "andOrHlt" ) ) return ( ! andOr_ ); // logically neutral, depending on base logical connective

  // Getting configuration parameters
  const vector< string > hltLogicalExpressions( config.getParameter< vector< string > >( "hltPaths" ) );
  hltInputTag_   = config.getParameter< InputTag >( "hltInputTag" );
  errorReplyHlt_ = config.getParameter< bool >( "errorReplyHlt" );

  // An empty HLT logical expressions list acts as switch.
  if ( hltLogicalExpressions.empty() ) return ( ! andOr_ );

  // Checking the TriggerResults InputTag
  // The process name has to be given.
  if ( hltInputTag_.process().size() == 0 ) {
    LogError( "hltProcess" ) << "HLT TriggerResults InputTag " << hltInputTag_.encode() << " specifies no process ==> decision: " << errorReplyHlt_;
    return errorReplyHlt_;
  }

  // Checking the HLT configuration,
  // initialization already in the calling DQM module (beginRun())
  if ( ! hltConfigInit ) {
    LogError( "hltConfigInit" ) << "HLT config initialization error with process name " << hltInputTag_.process() << " ==> decision: " << errorReplyHlt_;
    return errorReplyHlt_;
  }
  if ( hltConfig.size() <= 0 ) {
    LogError( "hltConfigSize" ) << "HLT config size error ==> decision: " << errorReplyHlt_;
    return errorReplyHlt_;
  }

  // Accessing the TriggerResults
  event.getByLabel( hltInputTag_, hltTriggerResults_ );
  if ( ! hltTriggerResults_.isValid() ) {
    LogError( "triggerResultsValid" ) << "TriggerResults product with InputTag " << hltInputTag_.encode() << " not in event ==> decision: " << errorReplyHlt_;
    return errorReplyHlt_;
  }

  // Determine decision of HLT logical expression combination and return
  if ( config.getParameter< bool >( "andOrHlt" ) ) { // OR combination
    for ( vector< string >::const_iterator hltLogicalExpression = hltLogicalExpressions.begin(); hltLogicalExpression != hltLogicalExpressions.end(); ++hltLogicalExpression ) {
      if ( acceptHltLogicalExpression( *hltLogicalExpression, hltConfig ) ) return true;
    }
    return false;
  }
  for ( vector< string >::const_iterator hltLogicalExpression = hltLogicalExpressions.begin(); hltLogicalExpression != hltLogicalExpressions.end(); ++hltLogicalExpression ) {
    if ( ! acceptHltLogicalExpression( *hltLogicalExpression, hltConfig ) ) return false;
  }
  return true;

}


/// Was this event accepted by this particular HLT paths' logical expression?
bool TriggerHelper::acceptHltLogicalExpression( string hltLogicalExpression, const HLTConfigProvider & hltConfig ) const
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
    const unsigned indexPath( hltConfig.triggerIndex( hltPathName ) );
    // Further error checks
    if ( indexPath == hltConfig.size() ) {
      LogError( "hltPathInProcess" ) << "HLT path " << hltPathName << " is not found in process " << hltInputTag_.process() << " ==> decision: " << errorReplyHlt_;
      hltAlgoLogicParser.operandTokenVector().at( iPath ).tokenResult = errorReplyHlt_;
      continue;
    }
    if ( hltTriggerResults_->error( indexPath ) ) {
      LogError( "hltPathError" ) << "HLT path " << hltPathName << " in error ==> decision: " << errorReplyHlt_;
      hltAlgoLogicParser.operandTokenVector().at( iPath ).tokenResult = errorReplyHlt_;
      continue;
    }
    // Manipulate algo decision as stored in the parser
    const bool decision( hltTriggerResults_->accept( indexPath ) );
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
