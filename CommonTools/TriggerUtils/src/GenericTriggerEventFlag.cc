//
// $Id: GenericTriggerEventFlag.cc,v 1.13 2012/04/22 15:09:29 vadler Exp $
//


#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"


// Constants' definitions
static const bool useL1EventSetup( true );
static const bool useL1GtTriggerMenuLite( false );


/// To be called from the ED module's c'tor
GenericTriggerEventFlag::GenericTriggerEventFlag( const edm::ParameterSet & config, edm::ConsumesCollector & iC )
  : watchDB_( 0 )
  , hltConfigInit_( false )
  , andOr_( false )
  , dbLabel_( "" )
  , verbose_( 0 )
  , andOrDcs_( false )
  , errorReplyDcs_( false )
  , andOrGt_( false )
  , gtInputTag_( "" )
  , gtEvmInputTag_( "" )
  , gtDBKey_( "" )
  , errorReplyGt_( false )
  , andOrL1_( false )
  , l1BeforeMask_( true )
  , l1DBKey_( "" )
  , errorReplyL1_( false )
  , andOrHlt_( false )
  , hltDBKey_( "" )
  , errorReplyHlt_( false )
  , on_( true )
  , onDcs_( true )
  , onGt_( true )
  , onL1_( true )
  , onHlt_( true )
  , configError_( "CONFIG_ERROR" )
  , emptyKeyError_( "EMPTY_KEY_ERROR" )
{

  // General switch(es)
  if ( config.exists( "andOr" ) ) {
    andOr_ = config.getParameter< bool >( "andOr" );
    if ( config.exists( "verbosityLevel" ) ) verbose_ = config.getParameter< unsigned >( "verbosityLevel" );
  } else {
    on_    = false;
    onDcs_ = false;
    onGt_  = false;
    onL1_  = false;
    onHlt_ = false;
  }

  if ( on_ ) {
    if ( config.exists( "andOrDcs" ) ) {
      andOrDcs_      = config.getParameter< bool >( "andOrDcs" );
      dcsInputTag_   = config.getParameter< edm::InputTag >( "dcsInputTag" );
      dcsInputToken_ = iC.mayConsume< DcsStatusCollection >( dcsInputTag_ );
      dcsPartitions_ = config.getParameter< std::vector< int > >( "dcsPartitions" );
      errorReplyDcs_ = config.getParameter< bool >( "errorReplyDcs" );
    } else {
      onDcs_ = false;
    }
    if ( config.exists( "andOrGt" ) ) {
      andOrGt_              = config.getParameter< bool >( "andOrGt" );
      gtInputTag_           = config.getParameter< edm::InputTag >( "gtInputTag" );
      gtInputToken_         = iC.mayConsume< L1GlobalTriggerReadoutRecord >( gtInputTag_ );
      gtLogicalExpressions_ = config.getParameter< std::vector< std::string > >( "gtStatusBits" );
      errorReplyGt_         = config.getParameter< bool >( "errorReplyGt" );
      if ( config.exists( "gtEvmInputTag" ) ) {
        gtEvmInputTag_   = config.getParameter< edm::InputTag >( "gtEvmInputTag" );
        gtEvmInputToken_ = iC.mayConsume< L1GlobalTriggerEvmReadoutRecord >( gtEvmInputTag_ );
      }
      if ( config.exists( "gtDBKey" ) ) gtDBKey_ = config.getParameter< std::string >( "gtDBKey" );
    } else {
      onGt_ = false;
    }
    if ( config.exists( "andOrL1" ) ) {
      andOrL1_                   = config.getParameter< bool >( "andOrL1" );
      l1LogicalExpressionsCache_ = config.getParameter< std::vector< std::string > >( "l1Algorithms" );
      errorReplyL1_              = config.getParameter< bool >( "errorReplyL1" );
      if ( config.exists( "l1DBKey" ) )      l1DBKey_      = config.getParameter< std::string >( "l1DBKey" );
      if ( config.exists( "l1BeforeMask" ) ) l1BeforeMask_ = config.getParameter< bool >( "l1BeforeMask" );
    } else {
      onL1_ = false;
    }
    if ( config.exists( "andOrHlt" ) ) {
      andOrHlt_                   = config.getParameter< bool >( "andOrHlt" );
      hltInputTag_                = config.getParameter< edm::InputTag >( "hltInputTag" );
      hltInputToken_              = iC.mayConsume< edm::TriggerResults >( hltInputTag_ );
      hltLogicalExpressionsCache_ = config.getParameter< std::vector< std::string > >( "hltPaths" );
      errorReplyHlt_              = config.getParameter< bool >( "errorReplyHlt" );
      if ( config.exists( "hltDBKey" ) ) hltDBKey_ = config.getParameter< std::string >( "hltDBKey" );
    } else {
      onHlt_ = false;
    }
    if ( ! onDcs_ && ! onGt_ && ! onL1_ && ! onHlt_ ) on_ = false;
    else {
      if ( config.exists( "dbLabel" ) ) dbLabel_ = config.getParameter< std::string >( "dbLabel" );
      watchDB_ = new edm::ESWatcher< AlCaRecoTriggerBitsRcd >;
    }
  }

}


/// To be called from d'tors by 'delete'
GenericTriggerEventFlag::~GenericTriggerEventFlag()
{

  if ( on_ ) delete watchDB_;

}


/// To be called from beginRun() methods
void GenericTriggerEventFlag::initRun( const edm::Run & run, const edm::EventSetup & setup )
{

  if ( watchDB_->check( setup ) ) {
    if ( onGt_ && gtDBKey_.size() > 0 ) {
      const std::vector< std::string > exprs( expressionsFromDB( gtDBKey_, setup ) );
      if ( exprs.empty() || exprs.at( 0 ) != configError_ ) gtLogicalExpressions_ = exprs;
    }
    if ( onL1_ && l1DBKey_.size() > 0 ) {
      const std::vector< std::string > exprs( expressionsFromDB( l1DBKey_, setup ) );
      if ( exprs.empty() || exprs.at( 0 ) != configError_ ) l1LogicalExpressionsCache_ = exprs;
    }
    if ( onHlt_ && hltDBKey_.size() > 0 ) {
      const std::vector< std::string > exprs( expressionsFromDB( hltDBKey_, setup ) );
      if ( exprs.empty() || exprs.at( 0 ) != configError_ ) hltLogicalExpressionsCache_ = exprs;
    }
  }

  // Re-initialise starting valuse before wild-card expansion
  l1LogicalExpressions_  = l1LogicalExpressionsCache_;
  hltLogicalExpressions_ = hltLogicalExpressionsCache_;

  hltConfigInit_ = false;
  if ( onHlt_ ) {
    if ( hltInputTag_.process().size() == 0 ) {
      if ( verbose_ > 0 ) edm::LogError( "GenericTriggerEventFlag" ) << "HLT TriggerResults InputTag \"" << hltInputTag_.encode() << "\" specifies no process";
    } else {
      bool hltChanged( false );
      if ( ! hltConfig_.init( run, setup, hltInputTag_.process(), hltChanged ) ) {
        if ( verbose_ > 0 ) edm::LogError( "GenericTriggerEventFlag" ) << "HLT config initialization error with process name \"" << hltInputTag_.process() << "\"";
      } else if ( hltConfig_.size() <= 0 ) {
        if ( verbose_ > 0 ) edm::LogError( "GenericTriggerEventFlag" ) << "HLT config size error";
      } else hltConfigInit_ = true;
    }
  }

  // Expand version wild-cards in HLT logical expressions
  // L1
  if ( onL1_ ) {
    // build vector of algo names
    l1Gt_->getL1GtRunCache( run, setup, useL1EventSetup, useL1GtTriggerMenuLite );
    edm::ESHandle< L1GtTriggerMenu > handleL1GtTriggerMenu;
    setup.get< L1GtTriggerMenuRcd >().get( handleL1GtTriggerMenu );
//     L1GtTriggerMenu l1GtTriggerMenu( *handleL1GtTriggerMenu );
    std::vector< std::string > algoNames;
//     const AlgorithmMap l1GtPhys( l1GtTriggerMenu.gtAlgorithmMap() );
    const AlgorithmMap l1GtPhys( handleL1GtTriggerMenu->gtAlgorithmMap() );
    for ( CItAlgo iAlgo = l1GtPhys.begin(); iAlgo != l1GtPhys.end(); ++iAlgo ) {
      algoNames.push_back( iAlgo->second.algoName() );
    }
//     const AlgorithmMap l1GtTech( l1GtTriggerMenu.gtTechnicalTriggerMap() );
    const AlgorithmMap l1GtTech( handleL1GtTriggerMenu->gtTechnicalTriggerMap() );
    for ( CItAlgo iAlgo = l1GtTech.begin(); iAlgo != l1GtTech.end(); ++iAlgo ) {
      algoNames.push_back( iAlgo->second.algoName() );
    }
    for ( unsigned iExpr = 0; iExpr < l1LogicalExpressions_.size(); ++iExpr ) {
      std::string l1LogicalExpression( l1LogicalExpressions_.at( iExpr ) );
      L1GtLogicParser l1AlgoLogicParser( l1LogicalExpression );
      // Loop over algorithms
      for ( size_t iAlgo = 0; iAlgo < l1AlgoLogicParser.operandTokenVector().size(); ++iAlgo ) {
        const std::string l1AlgoName( l1AlgoLogicParser.operandTokenVector().at( iAlgo ).tokenName );
        if ( l1AlgoName.find( '*' ) != std::string::npos ) {
          l1LogicalExpression.replace( l1LogicalExpression.find( l1AlgoName ), l1AlgoName.size(), expandLogicalExpression( algoNames, l1AlgoName ) );
        }
      }
      l1LogicalExpressions_[ iExpr ] = l1LogicalExpression;
    }
  }
  // HLT
  if ( hltConfigInit_ ) {
    for ( unsigned iExpr = 0; iExpr < hltLogicalExpressions_.size(); ++iExpr ) {
      std::string hltLogicalExpression( hltLogicalExpressions_.at( iExpr ) );
      L1GtLogicParser hltAlgoLogicParser( hltLogicalExpression );
      // Loop over paths
      for ( size_t iPath = 0; iPath < hltAlgoLogicParser.operandTokenVector().size(); ++iPath ) {
        const std::string hltPathName( hltAlgoLogicParser.operandTokenVector().at( iPath ).tokenName );
        if ( hltPathName.find( '*' ) != std::string::npos ) {
          hltLogicalExpression.replace( hltLogicalExpression.find( hltPathName ), hltPathName.size(), expandLogicalExpression( hltConfig_.triggerNames(), hltPathName ) );
        }
      }
      hltLogicalExpressions_[ iExpr ] = hltLogicalExpression;
    }
  }

}


/// To be called from analyze/filter() methods
bool GenericTriggerEventFlag::accept( const edm::Event & event, const edm::EventSetup & setup )
{

  if ( ! on_ ) return true;

  // Determine decision
  if ( andOr_ ) return ( acceptDcs( event ) || acceptGt( event ) || acceptL1( event, setup ) || acceptHlt( event ) );
  return ( acceptDcs( event ) && acceptGt( event ) && acceptL1( event, setup ) && acceptHlt( event ) );

}


bool GenericTriggerEventFlag::acceptDcs( const edm::Event & event )
{

  // An empty DCS partitions list acts as switch.
  if ( ! onDcs_ || dcsPartitions_.empty() ) return ( ! andOr_ ); // logically neutral, depending on base logical connective

  // Accessing the DcsStatusCollection
  edm::Handle< DcsStatusCollection > dcsStatus;
  event.getByToken( dcsInputToken_, dcsStatus );
  if ( ! dcsStatus.isValid() ) {
    if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "DcsStatusCollection product with InputTag \"" << dcsInputTag_.encode() << "\" not in event ==> decision: " << errorReplyDcs_;
    return errorReplyDcs_;
  }
  if ( ( *dcsStatus ).size() == 0 ) {
    if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "DcsStatusCollection product with InputTag \"" << dcsInputTag_.encode() << "\" empty ==> decision: " << errorReplyDcs_;
    return errorReplyDcs_;
  }

  // Determine decision of DCS partition combination and return
  if ( andOrDcs_ ) { // OR combination
    for ( std::vector< int >::const_iterator partitionNumber = dcsPartitions_.begin(); partitionNumber != dcsPartitions_.end(); ++partitionNumber ) {
      if ( acceptDcsPartition( dcsStatus, *partitionNumber ) ) return true;
    }
    return false;
  }
  for ( std::vector< int >::const_iterator partitionNumber = dcsPartitions_.begin(); partitionNumber != dcsPartitions_.end(); ++partitionNumber ) {
    if ( ! acceptDcsPartition( dcsStatus, *partitionNumber ) ) return false;
  }
  return true;

}


bool GenericTriggerEventFlag::acceptDcsPartition( const edm::Handle< DcsStatusCollection > & dcsStatus, int dcsPartition ) const
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
      if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "DCS partition number \"" << dcsPartition << "\" does not exist ==> decision: " << errorReplyDcs_;
      return errorReplyDcs_;
  }

  // Determine decision
  return dcsStatus->at( 0 ).ready( dcsPartition );

}


/// Does this event fulfill the configured GT status logical expression combination?
bool GenericTriggerEventFlag::acceptGt( const edm::Event & event )
{

  // An empty GT status bits logical expressions list acts as switch.
  if ( ! onGt_ || gtLogicalExpressions_.empty() ) return ( ! andOr_ ); // logically neutral, depending on base logical connective

  // Determine decision of GT status bits logical expression combination and return
  if ( andOrGt_ ) { // OR combination
    for ( std::vector< std::string >::const_iterator gtLogicalExpression = gtLogicalExpressions_.begin(); gtLogicalExpression != gtLogicalExpressions_.end(); ++gtLogicalExpression ) {
      if ( acceptGtLogicalExpression( event, *gtLogicalExpression ) ) return true;
    }
    return false;
  }
  for ( std::vector< std::string >::const_iterator gtLogicalExpression = gtLogicalExpressions_.begin(); gtLogicalExpression != gtLogicalExpressions_.end(); ++gtLogicalExpression ) {
    if ( ! acceptGtLogicalExpression( event, *gtLogicalExpression ) ) return false;
  }
  return true;

}


/// Does this event fulfill this particular GT status bits' logical expression?
bool GenericTriggerEventFlag::acceptGtLogicalExpression( const edm::Event & event, std::string gtLogicalExpression )
{

  // Check empty std::strings
  if ( gtLogicalExpression.empty() ) {
    if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "Empty logical expression ==> decision: " << errorReplyGt_;
    return errorReplyGt_;
  }

  // Negated paths
  bool negExpr( negate( gtLogicalExpression ) );
  if ( negExpr && gtLogicalExpression.empty() ) {
    if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "Empty (negated) logical expression ==> decision: " << errorReplyGt_;
    return errorReplyGt_;
  }

  // Parse logical expression and determine GT status bit decision
  L1GtLogicParser gtAlgoLogicParser( gtLogicalExpression );
  // Loop over status bits
  for ( size_t iStatusBit = 0; iStatusBit < gtAlgoLogicParser.operandTokenVector().size(); ++iStatusBit ) {
    const std::string gtStatusBit( gtAlgoLogicParser.operandTokenVector().at( iStatusBit ).tokenName );
    // Manipulate status bit decision as stored in the parser
    bool decision( errorReplyDcs_ );
    // Hard-coded status bits!!!
    if ( gtStatusBit == "PhysDecl" || gtStatusBit == "PhysicsDeclared" ) {
      edm::Handle< L1GlobalTriggerReadoutRecord > gtReadoutRecord;
      event.getByToken( gtInputToken_, gtReadoutRecord );
      if ( ! gtReadoutRecord.isValid() ) {
        if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "L1GlobalTriggerReadoutRecord product with InputTag \"" << gtInputTag_.encode() << "\" not in event ==> decision: " << errorReplyGt_;
        gtAlgoLogicParser.operandTokenVector().at( iStatusBit ).tokenResult = errorReplyDcs_;
        continue;
      }
      decision = ( gtReadoutRecord->gtFdlWord().physicsDeclared() == 1 );
    } else if ( gtStatusBit == "Stable" || gtStatusBit == "StableBeam" || gtStatusBit == "Adjust" || gtStatusBit == "Sqeeze" || gtStatusBit == "Flat" || gtStatusBit == "FlatTop" ||
                gtStatusBit == "7TeV" || gtStatusBit == "8TeV" || gtStatusBit == "2360GeV" || gtStatusBit == "900GeV" ) {
      edm::Handle< L1GlobalTriggerEvmReadoutRecord > gtEvmReadoutRecord;
      event.getByToken( gtEvmInputToken_, gtEvmReadoutRecord );
      if ( ! gtEvmReadoutRecord.isValid() ) {
        if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "L1GlobalTriggerEvmReadoutRecord product with InputTag \"" << gtEvmInputTag_.encode() << "\" not in event ==> decision: " << errorReplyGt_;
        gtAlgoLogicParser.operandTokenVector().at( iStatusBit ).tokenResult = errorReplyDcs_;
        continue;
      }
      if ( gtStatusBit == "Stable" || gtStatusBit == "StableBeam" ) {
        decision = ( gtEvmReadoutRecord->gtfeWord().beamMode() == 11 );
      } else if ( gtStatusBit == "Adjust" ) {
        decision = ( 10 <= gtEvmReadoutRecord->gtfeWord().beamMode() && gtEvmReadoutRecord->gtfeWord().beamMode() <= 11 );
      } else if ( gtStatusBit == "Sqeeze" ) {
        decision = ( 9 <= gtEvmReadoutRecord->gtfeWord().beamMode() && gtEvmReadoutRecord->gtfeWord().beamMode() <= 11 );
      } else if ( gtStatusBit == "Flat" || gtStatusBit == "FlatTop" ) {
        decision = ( 8 <= gtEvmReadoutRecord->gtfeWord().beamMode() && gtEvmReadoutRecord->gtfeWord().beamMode() <= 11 );
      } else if ( gtStatusBit == "7TeV" ) {
        decision = ( gtEvmReadoutRecord->gtfeWord().beamMomentum() == 3500 );
      } else if ( gtStatusBit == "8TeV" ) {
        decision = ( gtEvmReadoutRecord->gtfeWord().beamMomentum() == 4000 );
      } else if ( gtStatusBit == "2360GeV" ) {
        decision = ( gtEvmReadoutRecord->gtfeWord().beamMomentum() == 1180 );
      } else if ( gtStatusBit == "900GeV" ) {
        decision = ( gtEvmReadoutRecord->gtfeWord().beamMomentum() == 450 );
      }
    } else {
      if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "GT status bit \"" << gtStatusBit << "\" is not defined ==> decision: " << errorReplyGt_;
    }
    gtAlgoLogicParser.operandTokenVector().at( iStatusBit ).tokenResult = decision;
  }

  // Determine decision
  const bool gtDecision( gtAlgoLogicParser.expressionResult() );
  return negExpr ? ( ! gtDecision ) : gtDecision;

}


/// Was this event accepted by the configured L1 logical expression combination?
bool GenericTriggerEventFlag::acceptL1( const edm::Event & event, const edm::EventSetup & setup )
{

  // An empty L1 logical expressions list acts as switch.
  if ( ! onL1_ || l1LogicalExpressions_.empty() ) return ( ! andOr_ ); // logically neutral, depending on base logical connective

  // Getting the L1 event setup
  l1Gt_->getL1GtRunCache( event, setup, useL1EventSetup, useL1GtTriggerMenuLite ); // FIXME This can possibly go to initRun()

  // Determine decision of L1 logical expression combination and return
  if ( andOrL1_ ) { // OR combination
    for ( std::vector< std::string >::const_iterator l1LogicalExpression = l1LogicalExpressions_.begin(); l1LogicalExpression != l1LogicalExpressions_.end(); ++l1LogicalExpression ) {
      if ( acceptL1LogicalExpression( event, *l1LogicalExpression ) ) return true;
    }
    return false;
  }
  for ( std::vector< std::string >::const_iterator l1LogicalExpression = l1LogicalExpressions_.begin(); l1LogicalExpression != l1LogicalExpressions_.end(); ++l1LogicalExpression ) {
    if ( ! acceptL1LogicalExpression( event, *l1LogicalExpression ) ) return false;
  }
  return true;

}


/// Was this event accepted by this particular L1 algorithms' logical expression?
bool GenericTriggerEventFlag::acceptL1LogicalExpression( const edm::Event & event, std::string l1LogicalExpression )
{

  // Check empty std::strings
  if ( l1LogicalExpression.empty() ) {
    if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "Empty logical expression ==> decision: " << errorReplyL1_;
    return errorReplyL1_;
  }

  // Negated logical expression
  bool negExpr( negate( l1LogicalExpression ) );
  if ( negExpr && l1LogicalExpression.empty() ) {
    if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "Empty (negated) logical expression ==> decision: " << errorReplyL1_;
    return errorReplyL1_;
  }

  // Parse logical expression and determine L1 decision
  L1GtLogicParser l1AlgoLogicParser( l1LogicalExpression );
  // Loop over algorithms
  for ( size_t iAlgorithm = 0; iAlgorithm < l1AlgoLogicParser.operandTokenVector().size(); ++iAlgorithm ) {
    const std::string l1AlgoName( l1AlgoLogicParser.operandTokenVector().at( iAlgorithm ).tokenName );
    int error( -1 );
    const bool decision( l1BeforeMask_ ? l1Gt_->decisionBeforeMask( event, l1AlgoName, error ) : l1Gt_->decisionAfterMask( event, l1AlgoName, error ) );
    // Error checks
    if ( error != 0 ) {
      if ( verbose_ > 1 ) {
        if ( error == 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "L1 algorithm \"" << l1AlgoName << "\" does not exist in the L1 menu ==> decision: "                                          << errorReplyL1_;
        else              edm::LogWarning( "GenericTriggerEventFlag" ) << "L1 algorithm \"" << l1AlgoName << "\" received error code " << error << " from L1GtUtils::decisionBeforeMask ==> decision: " << errorReplyL1_;
      }
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
bool GenericTriggerEventFlag::acceptHlt( const edm::Event & event )
{

  // An empty HLT logical expressions list acts as switch.
  if ( ! onHlt_ || hltLogicalExpressions_.empty() ) return ( ! andOr_ ); // logically neutral, depending on base logical connective

  // Checking the HLT configuration,
  if ( ! hltConfigInit_ ) {
    if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "HLT config error ==> decision: " << errorReplyHlt_;
    return errorReplyHlt_;
  }

  // Accessing the TriggerResults
  edm::Handle< edm::TriggerResults > hltTriggerResults;
  event.getByToken( hltInputToken_, hltTriggerResults );
  if ( ! hltTriggerResults.isValid() ) {
    if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "TriggerResults product with InputTag \"" << hltInputTag_.encode() << "\" not in event ==> decision: " << errorReplyHlt_;
    return errorReplyHlt_;
  }
  if ( ( *hltTriggerResults ).size() == 0 ) {
    if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "TriggerResults product with InputTag \"" << hltInputTag_.encode() << "\" empty ==> decision: " << errorReplyHlt_;
    return errorReplyDcs_;
  }

  // Determine decision of HLT logical expression combination and return
  if ( andOrHlt_ ) { // OR combination
    for ( std::vector< std::string >::const_iterator hltLogicalExpression = hltLogicalExpressions_.begin(); hltLogicalExpression != hltLogicalExpressions_.end(); ++hltLogicalExpression ) {
      if ( acceptHltLogicalExpression( hltTriggerResults, *hltLogicalExpression ) ) return true;
    }
    return false;
  }
  for ( std::vector< std::string >::const_iterator hltLogicalExpression = hltLogicalExpressions_.begin(); hltLogicalExpression != hltLogicalExpressions_.end(); ++hltLogicalExpression ) {
    if ( ! acceptHltLogicalExpression( hltTriggerResults, *hltLogicalExpression ) ) return false;
  }
  return true;

}


/// Was this event accepted by this particular HLT paths' logical expression?
bool GenericTriggerEventFlag::acceptHltLogicalExpression( const edm::Handle< edm::TriggerResults > & hltTriggerResults, std::string hltLogicalExpression ) const
{

  // Check empty std::strings
  if ( hltLogicalExpression.empty() ) {
    if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "Empty logical expression ==> decision: " << errorReplyHlt_;
    return errorReplyHlt_;
  }

  // Negated paths
  bool negExpr( negate( hltLogicalExpression ) );
  if ( negExpr && hltLogicalExpression.empty() ) {
    if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "Empty (negated) logical expression ==> decision: " << errorReplyHlt_;
    return errorReplyHlt_;
  }

  // Parse logical expression and determine HLT decision
  L1GtLogicParser hltAlgoLogicParser( hltLogicalExpression );
  // Loop over paths
  for ( size_t iPath = 0; iPath < hltAlgoLogicParser.operandTokenVector().size(); ++iPath ) {
    const std::string hltPathName( hltAlgoLogicParser.operandTokenVector().at( iPath ).tokenName );
    const unsigned indexPath( hltConfig_.triggerIndex( hltPathName ) );
    // Further error checks
    if ( indexPath == hltConfig_.size() ) {
      if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "HLT path \"" << hltPathName << "\" is not found in process " << hltInputTag_.process() << " ==> decision: " << errorReplyHlt_;
      hltAlgoLogicParser.operandTokenVector().at( iPath ).tokenResult = errorReplyHlt_;
      continue;
    }
    if ( hltTriggerResults->error( indexPath ) ) {
      if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "HLT path \"" << hltPathName << "\" in error ==> decision: " << errorReplyHlt_;
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



/// Expand wild-carded logical expressions, giving version postfixes priority
std::string GenericTriggerEventFlag::expandLogicalExpression( const std::vector< std::string > & targets, const std::string & expr, bool useAnd ) const
{

  // Find matching entries in the menu
  std::vector< std::string > matched;
  const std::string versionWildcard( "_v*" );
  if ( expr.substr( expr.size() - versionWildcard.size() ) == versionWildcard ) {
    const std::string exprBase( expr.substr( 0, expr.size() - versionWildcard.size() ) );
    matched = hltConfig_.restoreVersion( targets, exprBase );
  } else {
    matched = hltConfig_.matched( targets, expr );
  }

  // Return input, if no match is found
  if ( matched.empty() ) {
    if ( verbose_ > 1 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "Logical expression: \"" << expr << "\" could not be resolved";
    return expr;
  }

  // Compose logical expression
  std::string expanded( "(" );
  for ( unsigned iVers = 0; iVers < matched.size(); ++iVers ) {
    if ( iVers > 0 ) expanded.append( useAnd ? " AND " : " OR " );
    expanded.append( matched.at( iVers ) );
  }
  expanded.append( ")" );
  if ( verbose_ > 1 ) edm::LogInfo( "GenericTriggerEventFlag" ) << "Logical expression: \"" << expr     << "\"\n"
                                                                << "   --> expanded to  \"" << expanded << "\"";

  return expanded;

}



/// Checks for negated words
bool GenericTriggerEventFlag::negate( std::string & word ) const
{

  bool negate( false );
  if ( word.at( 0 ) == '~' ) {
    negate = true;
    word.erase( 0, 1 );
  }
  return negate;

}



/// Reads and returns logical expressions from DB
std::vector< std::string > GenericTriggerEventFlag::expressionsFromDB( const std::string & key, const edm::EventSetup & setup )
{

  if ( key.size() == 0 ) return std::vector< std::string >( 1, emptyKeyError_ );
  edm::ESHandle< AlCaRecoTriggerBits > logicalExpressions;
  std::vector< edm::eventsetup::DataKey > labels;
  setup.get< AlCaRecoTriggerBitsRcd >().fillRegisteredDataKeys( labels );
  std::vector< edm::eventsetup::DataKey >::const_iterator iKey = labels.begin();
  while ( iKey != labels.end() && iKey->name().value() != dbLabel_ ) ++iKey;
  if ( iKey == labels.end() ) {
    if ( verbose_ > 0 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "Label " << dbLabel_ << " not found in DB for 'AlCaRecoTriggerBitsRcd'";
    return std::vector< std::string >( 1, configError_ );
  }
  setup.get< AlCaRecoTriggerBitsRcd >().get( dbLabel_, logicalExpressions );
  const std::map< std::string, std::string > & expressionMap = logicalExpressions->m_alcarecoToTrig;
  std::map< std::string, std::string >::const_iterator listIter = expressionMap.find( key );
  if ( listIter == expressionMap.end() ) {
    if ( verbose_ > 0 ) edm::LogWarning( "GenericTriggerEventFlag" ) << "No logical expressions found under key " << key << " in 'AlCaRecoTriggerBitsRcd'";
    return std::vector< std::string >( 1, configError_ );
  }
  return logicalExpressions->decompose( listIter->second );

}
