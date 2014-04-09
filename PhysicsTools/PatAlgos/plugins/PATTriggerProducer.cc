//
//


#include "PhysicsTools/PatAlgos/plugins/PATTriggerProducer.h"

#include <vector>
#include <map>
#include <utility>
#include <cassert>
#include <string>

#include "FWCore/Framework/interface/InputTagMatch.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/PatCandidates/interface/TriggerAlgorithm.h"
#include "DataFormats/PatCandidates/interface/TriggerCondition.h"
#include "DataFormats/PatCandidates/interface/TriggerPath.h"
#include "DataFormats/PatCandidates/interface/TriggerFilter.h"
#include "DataFormats/PatCandidates/interface/PackedTriggerPrescales.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace pat;
using namespace edm;


// Constants' definitions
const unsigned L1GlobalTriggerReadoutSetup::NumberPhysTriggers;
const unsigned L1GlobalTriggerReadoutSetup::NumberPhysTriggersExtended;
const unsigned L1GlobalTriggerReadoutSetup::NumberTechnicalTriggers;
static const bool useL1EventSetup( true );
static const bool useL1GtTriggerMenuLite( false );


PATTriggerProducer::PATTriggerProducer( const ParameterSet & iConfig ) :
  nameProcess_( iConfig.getParameter< std::string >( "processName" ) ),
  autoProcessName_( nameProcess_ == "*" ),
  onlyStandAlone_( iConfig.getParameter< bool >( "onlyStandAlone" ) ),
  firstInRun_( true ),
  // L1 configuration parameters
  addL1Algos_( false ),
  tagL1GlobalTriggerObjectMaps_( "l1L1GtObjectMap" ),
  tagL1ExtraMu_(),
  tagL1ExtraNoIsoEG_(),
  tagL1ExtraIsoEG_(),
  tagL1ExtraCenJet_(),
  tagL1ExtraForJet_(),
  tagL1ExtraTauJet_(),
  tagL1ExtraETM_(),
  tagL1ExtraHTM_(),
  autoProcessNameL1ExtraMu_( false ),
  autoProcessNameL1ExtraNoIsoEG_( false ),
  autoProcessNameL1ExtraIsoEG_( false ),
  autoProcessNameL1ExtraCenJet_( false ),
  autoProcessNameL1ExtraForJet_( false ),
  autoProcessNameL1ExtraTauJet_( false ),
  autoProcessNameL1ExtraETM_( false ),
  autoProcessNameL1ExtraHTM_( false ),
  mainBxOnly_( true ),
  saveL1Refs_( false ),
  // HLTConfigProvider
  hltConfigInit_( false ),
  // HLT configuration parameters
  tagTriggerResults_( "TriggerResults" ),
  tagTriggerEvent_( "hltTriggerSummaryAOD" ),
  hltPrescaleLabel_(),
  labelHltPrescaleTable_(),
  hltPrescaleTableRun_(),
  hltPrescaleTableLumi_(),
  addPathModuleLabels_( false ),
  packPathNames_( iConfig.existsAs<bool>("packTriggerPathNames") ? iConfig.getParameter<bool>("packTriggerPathNames") : false ),
  packPrescales_( iConfig.existsAs<bool>("packTriggerPrescales") ? iConfig.getParameter<bool>("packTriggerPrescales") : true )
{

  // L1 configuration parameters
  if ( iConfig.exists( "addL1Algos" ) ) addL1Algos_ = iConfig.getParameter< bool >( "addL1Algos" );
  if ( iConfig.exists( "l1GlobalTriggerObjectMaps" ) ) tagL1GlobalTriggerObjectMaps_ = iConfig.getParameter< InputTag >( "l1GlobalTriggerObjectMaps" );
  l1GlobalTriggerObjectMapsToken_ = mayConsume< L1GlobalTriggerObjectMaps >( tagL1GlobalTriggerObjectMaps_ );
  if ( iConfig.exists( "l1ExtraMu" ) ) {
    tagL1ExtraMu_ = iConfig.getParameter< InputTag >( "l1ExtraMu" );
    if ( tagL1ExtraMu_.process() == "*" ) {
      if ( autoProcessName_ ) autoProcessNameL1ExtraMu_ = true;
      else                    tagL1ExtraMu_ = InputTag( tagL1ExtraMu_.label(), tagL1ExtraMu_.instance(), nameProcess_ );
    }
    l1ExtraMuGetter_ = GetterOfProducts< l1extra::L1MuonParticleCollection >( InputTagMatch( InputTag( tagL1ExtraMu_.label(), tagL1ExtraMu_.instance() ) ), this);
  }
  if ( iConfig.exists( "l1ExtraNoIsoEG" ) ) {
    tagL1ExtraNoIsoEG_ = iConfig.getParameter< InputTag >( "l1ExtraNoIsoEG" );
    if ( tagL1ExtraNoIsoEG_.process() == "*" ) {
      if ( autoProcessName_ ) autoProcessNameL1ExtraNoIsoEG_ = true;
      else                    tagL1ExtraNoIsoEG_ = InputTag( tagL1ExtraNoIsoEG_.label(), tagL1ExtraNoIsoEG_.instance(), nameProcess_ );
    }
    l1ExtraNoIsoEGGetter_ = GetterOfProducts< l1extra::L1EmParticleCollection >( InputTagMatch( InputTag( tagL1ExtraNoIsoEG_.label(), tagL1ExtraNoIsoEG_.instance() ) ), this);
  }
  if ( iConfig.exists( "l1ExtraIsoEG" ) ) {
    tagL1ExtraIsoEG_ = iConfig.getParameter< InputTag >( "l1ExtraIsoEG" );
    if ( tagL1ExtraIsoEG_.process() == "*" ) {
      if ( autoProcessName_ ) autoProcessNameL1ExtraIsoEG_ = true;
      else                    tagL1ExtraIsoEG_ = InputTag( tagL1ExtraIsoEG_.label(), tagL1ExtraIsoEG_.instance(), nameProcess_ );
    }
    l1ExtraIsoEGGetter_ = GetterOfProducts< l1extra::L1EmParticleCollection >( InputTagMatch( InputTag( tagL1ExtraIsoEG_.label(), tagL1ExtraIsoEG_.instance() ) ), this);
  }
  if ( iConfig.exists( "l1ExtraCenJet" ) ) {
    tagL1ExtraCenJet_ = iConfig.getParameter< InputTag >( "l1ExtraCenJet" );
    if ( tagL1ExtraCenJet_.process() == "*" ) {
      if ( autoProcessName_ ) autoProcessNameL1ExtraCenJet_ = true;
      else                    tagL1ExtraCenJet_ = InputTag( tagL1ExtraCenJet_.label(), tagL1ExtraCenJet_.instance(), nameProcess_ );
    }
    l1ExtraCenJetGetter_ = GetterOfProducts< l1extra::L1JetParticleCollection >( InputTagMatch( InputTag( tagL1ExtraCenJet_.label(), tagL1ExtraCenJet_.instance() ) ), this);
  }
  if ( iConfig.exists( "l1ExtraForJet" ) ) {
    tagL1ExtraForJet_ = iConfig.getParameter< InputTag >( "l1ExtraForJet" );
    if ( tagL1ExtraForJet_.process() == "*" ) {
      if ( autoProcessName_ ) autoProcessNameL1ExtraForJet_ = true;
      else                    tagL1ExtraForJet_ = InputTag( tagL1ExtraForJet_.label(), tagL1ExtraForJet_.instance(), nameProcess_ );
    }
    l1ExtraForJetGetter_ = GetterOfProducts< l1extra::L1JetParticleCollection >( InputTagMatch( InputTag( tagL1ExtraForJet_.label(), tagL1ExtraForJet_.instance() ) ), this);
  }
  if ( iConfig.exists( "l1ExtraTauJet" ) ) {
    tagL1ExtraTauJet_ = iConfig.getParameter< InputTag >( "l1ExtraTauJet" );
    if ( tagL1ExtraTauJet_.process() == "*" ) {
      if ( autoProcessName_ ) autoProcessNameL1ExtraTauJet_ = true;
      else                    tagL1ExtraTauJet_ = InputTag( tagL1ExtraTauJet_.label(), tagL1ExtraTauJet_.instance(), nameProcess_ );
    }
    l1ExtraTauJetGetter_ = GetterOfProducts< l1extra::L1JetParticleCollection >( InputTagMatch( InputTag( tagL1ExtraTauJet_.label(), tagL1ExtraTauJet_.instance() ) ), this);
  }
  if ( iConfig.exists( "l1ExtraETM" ) ) {
    tagL1ExtraETM_ = iConfig.getParameter< InputTag >( "l1ExtraETM" );
    if ( tagL1ExtraETM_.process() == "*" ) {
      if ( autoProcessName_ ) autoProcessNameL1ExtraETM_ = true;
      else                    tagL1ExtraETM_ = InputTag( tagL1ExtraETM_.label(), tagL1ExtraETM_.instance(), nameProcess_ );
    }
    l1ExtraETMGetter_ = GetterOfProducts< l1extra::L1EtMissParticleCollection >( InputTagMatch( InputTag( tagL1ExtraETM_.label(), tagL1ExtraETM_.instance() ) ), this);
  }
  if ( iConfig.exists( "l1ExtraHTM" ) ) {
    tagL1ExtraHTM_ = iConfig.getParameter< InputTag >( "l1ExtraHTM" );
    if ( tagL1ExtraHTM_.process() == "*" ) {
      if ( autoProcessName_ ) autoProcessNameL1ExtraHTM_ = true;
      else                    tagL1ExtraHTM_ = InputTag( tagL1ExtraHTM_.label(), tagL1ExtraHTM_.instance(), nameProcess_ );
    }
    l1ExtraHTMGetter_ = GetterOfProducts< l1extra::L1EtMissParticleCollection >( InputTagMatch( InputTag( tagL1ExtraHTM_.label(), tagL1ExtraHTM_.instance() ) ), this);
  }
  if ( iConfig.exists( "mainBxOnly" ) ) mainBxOnly_ = iConfig.getParameter< bool >( "mainBxOnly" );
  if ( iConfig.exists( "saveL1Refs" ) ) saveL1Refs_ = iConfig.getParameter< bool >( "saveL1Refs" );

  // HLT configuration parameters
  if ( iConfig.exists( "triggerResults" ) ) tagTriggerResults_ = iConfig.getParameter< InputTag >( "triggerResults" );
  triggerResultsGetter_ = GetterOfProducts< TriggerResults >( InputTagMatch( InputTag( tagTriggerResults_.label(), tagTriggerResults_.instance() ) ), this);
  if ( iConfig.exists( "triggerEvent" ) ) tagTriggerEvent_ = iConfig.getParameter< InputTag >( "triggerEvent" );
  triggerEventGetter_ = GetterOfProducts< trigger::TriggerEvent >( InputTagMatch( InputTag( tagTriggerEvent_.label(), tagTriggerEvent_.instance() ) ), this);
  if ( iConfig.exists( "hltPrescaleLabel" ) )    hltPrescaleLabel_      = iConfig.getParameter< std::string >( "hltPrescaleLabel" );
  if ( iConfig.exists( "hltPrescaleTable" ) ) {
    labelHltPrescaleTable_ = iConfig.getParameter< std::string >( "hltPrescaleTable" );
    hltPrescaleTableRunGetter_ = GetterOfProducts< trigger::HLTPrescaleTable >( InputTagMatch( InputTag( labelHltPrescaleTable_, "Run" ) ), this, InRun );
    hltPrescaleTableLumiGetter_ = GetterOfProducts< trigger::HLTPrescaleTable >( InputTagMatch( InputTag( labelHltPrescaleTable_, "Lumi" ) ), this, InLumi );
    hltPrescaleTableEventGetter_ = GetterOfProducts< trigger::HLTPrescaleTable >( InputTagMatch( InputTag( labelHltPrescaleTable_, "Event" ) ), this );
  }
  if ( iConfig.exists( "addPathModuleLabels" ) ) addPathModuleLabels_ = iConfig.getParameter< bool >( "addPathModuleLabels" );
  exludeCollections_.clear();
  if ( iConfig.exists( "exludeCollections" )  ) exludeCollections_ = iConfig.getParameter< std::vector< std::string > >( "exludeCollections" );

  callWhenNewProductsRegistered( [ this, &iConfig ]( BranchDescription const& bd ) {
    if ( iConfig.exists( "l1ExtraMu" ) ) l1ExtraMuGetter_( bd );
    if ( iConfig.exists( "l1ExtraNoIsoEG" ) ) l1ExtraNoIsoEGGetter_( bd );
    if ( iConfig.exists( "l1ExtraIsoEG" ) ) l1ExtraIsoEGGetter_( bd );
    if ( iConfig.exists( "l1ExtraCenJet" ) ) l1ExtraCenJetGetter_( bd );
    if ( iConfig.exists( "l1ExtraForJet" ) ) l1ExtraForJetGetter_( bd );
    if ( iConfig.exists( "l1ExtraTauJet" ) ) l1ExtraTauJetGetter_( bd );
    if ( iConfig.exists( "l1ExtraETM" ) ) l1ExtraETMGetter_( bd );
    if ( iConfig.exists( "l1ExtraHTM" ) ) l1ExtraHTMGetter_( bd );
    triggerResultsGetter_( bd );
    triggerEventGetter_( bd );
    if ( iConfig.exists( "hltPrescaleTable" ) ) {
      hltPrescaleTableRunGetter_( bd );
      hltPrescaleTableLumiGetter_( bd );
      hltPrescaleTableEventGetter_( bd );
    }
  } );

  if ( ! onlyStandAlone_ ) {
    produces< TriggerAlgorithmCollection >();
    produces< TriggerConditionCollection >();
    produces< TriggerPathCollection >();
    produces< TriggerFilterCollection >();
    produces< TriggerObjectCollection >();
  }
  if (packPrescales_) {
    produces< PackedTriggerPrescales >();
  }
  produces< TriggerObjectStandAloneCollection >();

}


void PATTriggerProducer::beginRun(const Run & iRun, const EventSetup & iSetup )
{

  // Initialize
  firstInRun_    = true;
  l1PSet_        = 0;
  hltConfigInit_ = false;

  // Initialize process name
  if ( autoProcessName_ ) {
    // reset
    nameProcess_ = "*";
    // determine process name from last run TriggerSummaryProducerAOD module in process history of input
    const ProcessHistory & processHistory( iRun.processHistory() );
    ProcessConfiguration processConfiguration;
    ParameterSet processPSet;
    // unbroken loop, which relies on time ordering (accepts the last found entry)
    for ( ProcessHistory::const_iterator iHist = processHistory.begin(); iHist != processHistory.end(); ++iHist ) {
      if ( processHistory.getConfigurationForProcess( iHist->processName(), processConfiguration )     &&
           pset::Registry::instance()->getMapped( processConfiguration.parameterSetID(), processPSet ) &&
           processPSet.exists( tagTriggerEvent_.label() )
         ) {
        nameProcess_ = iHist->processName();
        LogDebug( "autoProcessName" ) << "HLT process name '" << nameProcess_ << "' discovered";
      }
    }
    // terminate, if nothing is found
    if ( nameProcess_ == "*" ) {
      LogError( "autoProcessName" ) << "trigger::TriggerEvent product with label '" << tagTriggerEvent_.label() << "' not produced according to process history of input data\n"
                                    << "No trigger information produced";
      return;
    }
    LogInfo( "autoProcessName" ) << "HLT process name' " << nameProcess_ << "' used for PAT trigger information";
  }
  // adapt configuration of used input tags
  if ( tagTriggerResults_.process().empty() || tagTriggerResults_.process() == "*" ) {
    tagTriggerResults_ = InputTag( tagTriggerResults_.label(), tagTriggerResults_.instance(), nameProcess_ );
  } else if ( tagTriggerEvent_.process() != nameProcess_ ) {
    LogWarning( "inputTags" ) << "TriggerResults process name '" << tagTriggerResults_.process() << "' differs from HLT process name '" << nameProcess_ << "'";
  }
  if ( tagTriggerEvent_.process().empty() || tagTriggerEvent_.process()   == "*" ) {
    tagTriggerEvent_ = InputTag( tagTriggerEvent_.label(), tagTriggerEvent_.instance(), nameProcess_ );
  } else if ( tagTriggerEvent_.process() != nameProcess_ ) {
    LogWarning( "inputTags" ) << "TriggerEvent process name '" << tagTriggerEvent_.process() << "' differs from HLT process name '" << nameProcess_ << "'";
  }
  if ( autoProcessNameL1ExtraMu_ )      tagL1ExtraMu_      = InputTag( tagL1ExtraMu_.label()     , tagL1ExtraMu_.instance()     , nameProcess_ );
  if ( autoProcessNameL1ExtraNoIsoEG_ ) tagL1ExtraNoIsoEG_ = InputTag( tagL1ExtraNoIsoEG_.label(), tagL1ExtraNoIsoEG_.instance(), nameProcess_ );
  if ( autoProcessNameL1ExtraIsoEG_ )   tagL1ExtraIsoEG_   = InputTag( tagL1ExtraIsoEG_.label()  , tagL1ExtraIsoEG_.instance()  , nameProcess_ );
  if ( autoProcessNameL1ExtraCenJet_ )  tagL1ExtraCenJet_  = InputTag( tagL1ExtraCenJet_.label() , tagL1ExtraCenJet_.instance() , nameProcess_ );
  if ( autoProcessNameL1ExtraForJet_ )  tagL1ExtraForJet_  = InputTag( tagL1ExtraForJet_.label() , tagL1ExtraForJet_.instance() , nameProcess_ );
  if ( autoProcessNameL1ExtraTauJet_ )  tagL1ExtraTauJet_  = InputTag( tagL1ExtraTauJet_.label() , tagL1ExtraTauJet_.instance() , nameProcess_ );
  if ( autoProcessNameL1ExtraETM_ )     tagL1ExtraETM_     = InputTag( tagL1ExtraETM_.label()    , tagL1ExtraETM_.instance()    , nameProcess_ );
  if ( autoProcessNameL1ExtraHTM_ )     tagL1ExtraHTM_     = InputTag( tagL1ExtraHTM_.label()    , tagL1ExtraHTM_.instance()    , nameProcess_ );

  // Initialize HLTConfigProvider
  bool changed( true );
  if ( ! hltConfig_.init( iRun, iSetup, nameProcess_, changed ) ) {
    LogError( "hltConfig" ) << "HLT config extraction error with process name '" << nameProcess_ << "'";
  } else if ( hltConfig_.size() <= 0 ) {
    LogError( "hltConfig" ) << "HLT config size error";
  } else hltConfigInit_ = true;

  // Update mapping from filter names to path names
  if (hltConfigInit_ && changed) moduleLabelToPathAndFlags_.init( hltConfig_ );

  // Extract pre-scales
  if ( hltConfigInit_ ) {
    // Start empty
    hltPrescaleTableRun_ = trigger::HLTPrescaleTable();
    // Try run product, if configured
    if ( ! labelHltPrescaleTable_.empty() ) {
      Handle< trigger::HLTPrescaleTable > handleHltPrescaleTable;
      iRun.getByLabel( InputTag( labelHltPrescaleTable_, "Run", nameProcess_ ), handleHltPrescaleTable );
      if ( handleHltPrescaleTable.isValid() ) {
        hltPrescaleTableRun_ = trigger::HLTPrescaleTable( handleHltPrescaleTable->set(), handleHltPrescaleTable->labels(), handleHltPrescaleTable->table() );
      }
    }
  }

}


void PATTriggerProducer::beginLuminosityBlock(const LuminosityBlock & iLuminosityBlock, const EventSetup & iSetup )
{

  // Terminate, if auto process name determination failed
  if ( nameProcess_ == "*" ) return;

  // Extract pre-scales
  if ( hltConfigInit_ ) {
    // Start from run
    hltPrescaleTableLumi_ = trigger::HLTPrescaleTable( hltPrescaleTableRun_.set(), hltPrescaleTableRun_.labels(), hltPrescaleTableRun_.table() );
    // Try lumi product, if configured and available
    if ( ! labelHltPrescaleTable_.empty() ) {
      Handle< trigger::HLTPrescaleTable > handleHltPrescaleTable;
      iLuminosityBlock.getByLabel( InputTag( labelHltPrescaleTable_, "Lumi", nameProcess_ ), handleHltPrescaleTable );
      if ( handleHltPrescaleTable.isValid() ) {
        hltPrescaleTableLumi_ = trigger::HLTPrescaleTable( handleHltPrescaleTable->set(), handleHltPrescaleTable->labels(), handleHltPrescaleTable->table() );
      }
    }
  }

}


void PATTriggerProducer::produce( Event& iEvent, const EventSetup& iSetup )
{

  // Terminate, if auto process name determination failed
  if ( nameProcess_ == "*" ) return;

  std::auto_ptr< TriggerObjectCollection > triggerObjects( new TriggerObjectCollection() );
  std::auto_ptr< TriggerObjectStandAloneCollection > triggerObjectsStandAlone( new TriggerObjectStandAloneCollection() );
  std::auto_ptr< PackedTriggerPrescales > packedPrescales;

  // HLT

  // Get and check HLT event data
  Handle< trigger::TriggerEvent > handleTriggerEvent;
  iEvent.getByLabel( tagTriggerEvent_, handleTriggerEvent );
  Handle< TriggerResults > handleTriggerResults;
  iEvent.getByLabel( tagTriggerResults_, handleTriggerResults );
  bool goodHlt( hltConfigInit_ );
  if ( goodHlt ) {
    if( ! handleTriggerResults.isValid() ) {
      LogError( "triggerResultsValid" ) << "TriggerResults product with InputTag '" << tagTriggerResults_.encode() << "' not in event\n"
                                        << "No HLT information produced";
      goodHlt = false;
    } else if ( ! handleTriggerEvent.isValid() ) {
      LogError( "triggerEventValid" ) << "trigger::TriggerEvent product with InputTag '" << tagTriggerEvent_.encode() << "' not in event\n"
                                      << "No HLT information produced";
      goodHlt = false;
    }
  }

  // Produce HLT paths and determine status of modules

  if ( goodHlt ) {

    // Extract pre-scales
    // Start from lumi
    trigger::HLTPrescaleTable hltPrescaleTable( hltPrescaleTableLumi_.set(), hltPrescaleTableLumi_.labels(), hltPrescaleTableLumi_.table() );
    // Try event product, if configured and available
    if ( ! labelHltPrescaleTable_.empty() ) {
      Handle< trigger::HLTPrescaleTable > handleHltPrescaleTable;
      iEvent.getByLabel( InputTag( labelHltPrescaleTable_, "Event", nameProcess_ ), handleHltPrescaleTable );
      if ( handleHltPrescaleTable.isValid() ) {
        hltPrescaleTable = trigger::HLTPrescaleTable( handleHltPrescaleTable->set(), handleHltPrescaleTable->labels(), handleHltPrescaleTable->table() );
      }
    }
    // Try event setup, if no product
    if ( hltPrescaleTable.size() == 0 ) {
      if ( ! labelHltPrescaleTable_.empty() ) {
        LogWarning( "hltPrescaleInputTag" ) << "HLTPrescaleTable product with label '" << labelHltPrescaleTable_ << "' not found in process" << nameProcess_ << "\n"
                                            << "Using default from event setup";
      }
      if ( hltConfig_.prescaleSize() > 0 ) {
        if ( hltConfig_.prescaleSet( iEvent, iSetup ) != -1 ) {
          hltPrescaleTable = trigger::HLTPrescaleTable( hltConfig_.prescaleSet( iEvent, iSetup ), hltConfig_.prescaleLabels(), hltConfig_.prescaleTable() );
          LogDebug( "hltPrescaleTable" ) << "HLT prescale table found in event setup";
        } else {
          LogWarning( "hltPrescaleSet" ) << "HLTPrescaleTable from event setup has error";
        }
      }
    }
    unsigned set( hltPrescaleTable.set() );
    if ( hltPrescaleTable.size() > 0 ) {
      if ( hltPrescaleLabel_.size() > 0 ) {
        bool foundPrescaleLabel( false );
        for ( unsigned iLabel = 0; iLabel <  hltPrescaleTable.labels().size(); ++iLabel ) {
          if ( hltPrescaleTable.labels().at( iLabel ) == hltPrescaleLabel_ ) {
            set                = iLabel;
            foundPrescaleLabel = true;
            break;
          }
        }
        if ( ! foundPrescaleLabel ) {
          LogWarning( "hltPrescaleLabel" ) << "HLT prescale label '" << hltPrescaleLabel_ << "' not in prescale table\n"
                                           << "Using default";
        }
      }
    } else if ( iEvent.isRealData() ) {
      if ( ( labelHltPrescaleTable_.empty() && firstInRun_ ) || ! labelHltPrescaleTable_.empty() ) {
        LogError( "hltPrescaleTable" ) << "No HLT prescale table found\n"
                                       << "Using default empty table with all prescales 1";
      }
    }

    const unsigned sizePaths( hltConfig_.size() );
    const unsigned sizeFilters( handleTriggerEvent->sizeFilters() );
    const unsigned sizeObjects( handleTriggerEvent->sizeObjects() );

    std::map< std::string, int > moduleStates;

    if ( ! onlyStandAlone_ ) {
      std::auto_ptr< TriggerPathCollection > triggerPaths( new TriggerPathCollection() );
      triggerPaths->reserve( sizePaths );
      const std::vector<std::string> & pathNames = hltConfig_.triggerNames();
      for ( size_t indexPath = 0; indexPath < sizePaths; ++indexPath ) {
        const std::string & namePath = pathNames.at( indexPath );
        unsigned indexLastFilterPathModules( handleTriggerResults->index( indexPath ) + 1 );
        unsigned indexLastFilterFilters( sizeFilters );
        while ( indexLastFilterPathModules > 0 ) {
          --indexLastFilterPathModules;
          const std::string & labelLastFilterPathModules( hltConfig_.moduleLabel( indexPath, indexLastFilterPathModules ) );
          indexLastFilterFilters = handleTriggerEvent->filterIndex( InputTag( labelLastFilterPathModules, "", nameProcess_ ) );
          if ( indexLastFilterFilters < sizeFilters ) {
            if ( hltConfig_.moduleType( labelLastFilterPathModules ) == "HLTBool" ) continue;
            break;
          }
        }
        TriggerPath triggerPath( namePath, indexPath, hltConfig_.prescaleValue( set, namePath ), handleTriggerResults->wasrun( indexPath ), handleTriggerResults->accept( indexPath ), handleTriggerResults->error( indexPath ), indexLastFilterPathModules, hltConfig_.saveTagsModules( namePath ).size() );
        // add module names to path and states' map
        const unsigned sizeModulesPath( hltConfig_.size( indexPath ) );
        assert( indexLastFilterPathModules < sizeModulesPath );
        std::map< unsigned, std::string > indicesModules;
        for ( size_t iM = 0; iM < sizeModulesPath; ++iM ) {
          const std::string nameModule( hltConfig_.moduleLabel( indexPath, iM ) );
          if ( addPathModuleLabels_ ) {
            triggerPath.addModule( nameModule );
          }
          const unsigned indexFilter( handleTriggerEvent->filterIndex( InputTag( nameModule, "", nameProcess_ ) ) );
          if ( indexFilter < sizeFilters ) {
            triggerPath.addFilterIndex( indexFilter );
          }
          const unsigned slotModule( hltConfig_.moduleIndex( indexPath, nameModule ) );
          indicesModules.insert( std::pair< unsigned, std::string >( slotModule, nameModule ) );
        }
        // add L1 seeds
        const L1SeedCollection l1Seeds( hltConfig_.hltL1GTSeeds( namePath ) );
        for ( L1SeedCollection::const_iterator iSeed = l1Seeds.begin(); iSeed != l1Seeds.end(); ++iSeed ) {
          triggerPath.addL1Seed( *iSeed );
        }
        // store path
        triggerPaths->push_back( triggerPath );
        // cache module states to be used for the filters
        for ( std::map< unsigned, std::string >::const_iterator iM = indicesModules.begin(); iM != indicesModules.end(); ++iM ) {
          if ( iM->first < indexLastFilterPathModules ) {
            moduleStates[ iM->second ] = 1;
          } else if ( iM->first == indexLastFilterPathModules ) {
            moduleStates[ iM->second ] = handleTriggerResults->accept( indexPath );
          } else if ( moduleStates.find( iM->second ) == moduleStates.end() ) {
            moduleStates[ iM->second ] = -1;
          }
        }
      }
      // Put HLT paths to event
      iEvent.put( triggerPaths );
    }

    // Store used trigger objects and their types for HLT filters
    // (only active filter(s) available from trigger::TriggerEvent)

    std::multimap< trigger::size_type, int >         objectTypes;
    std::multimap< trigger::size_type, std::string > filterLabels;

    for ( size_t iF = 0; iF < sizeFilters; ++iF ) {
      const std::string nameFilter( handleTriggerEvent->filterLabel( iF ) );
      const trigger::Keys & keys  = handleTriggerEvent->filterKeys( iF );
      const trigger::Vids & types = handleTriggerEvent->filterIds( iF );
      assert( types.size() == keys.size() );
      for ( size_t iK = 0; iK < keys.size(); ++iK ) {
        filterLabels.insert( std::pair< trigger::size_type, std::string >( keys[ iK ], nameFilter ) );
        objectTypes.insert( std::pair< trigger::size_type, int >( keys[ iK ], types[ iK ] ) );
      }
    }

    // HLT objects

    triggerObjects->reserve( onlyStandAlone_ ? 0 : sizeObjects );
    triggerObjectsStandAlone->reserve( sizeObjects );

    const trigger::Keys & collectionKeys( handleTriggerEvent->collectionKeys() );
    std::map< trigger::size_type, trigger::size_type > newObjectKeys;
    for ( size_t iO = 0, iC = 0, nC = handleTriggerEvent->sizeCollections(); iO < sizeObjects && iC < nC; ++iO ) {
      const trigger::TriggerObject tobj = handleTriggerEvent->getObjects().at( iO );
      TriggerObject triggerObject( reco::Particle::PolarLorentzVector(tobj.pt(), tobj.eta(), tobj.phi(), tobj.mass()), tobj.id()  );
      // set collection
      while ( iO >= collectionKeys[ iC ] ) ++iC; // relies on well ordering of trigger objects with respect to the collections
      triggerObject.setCollection( handleTriggerEvent->collectionTagEncoded( iC ) );
      // set filter ID
      typedef std::multimap< trigger::size_type, int >::const_iterator it_type;
      for (std::pair<it_type,it_type> trange = objectTypes.equal_range(iO);
          trange.first != trange.second; ++trange.first) {
          triggerObject.addTriggerObjectType( trange.first->second );
      }

      // stand-alone trigger object
      TriggerObjectStandAlone triggerObjectStandAlone( triggerObject );
      // check for excluded collections
      bool excluded( false );
      for ( size_t iE = 0; iE < exludeCollections_.size(); ++iE ) {
        if ( triggerObjectStandAlone.hasCollection( exludeCollections_.at( iE ) ) ) {
          if ( ! onlyStandAlone_ ) newObjectKeys[ iO ] = trigger::size_type( sizeObjects );
          excluded = true;
          break;
        }
      }
      if ( excluded ) continue;
      typedef std::multimap< trigger::size_type, std::string >::const_iterator it_fl;
      for (std::pair<it_fl,it_fl> frange = filterLabels.equal_range(iO); frange.first != frange.second; ++frange.first) {
          triggerObjectStandAlone.addFilterLabel( frange.first->second );
          const std::vector<ModuleLabelToPathAndFlags::PathAndFlags> & paths = moduleLabelToPathAndFlags_[frange.first->second];
          for (std::vector<ModuleLabelToPathAndFlags::PathAndFlags>::const_iterator iP = paths.begin(); iP != paths.end(); ++iP) {
              bool pathFired = handleTriggerResults->wasrun( iP->pathIndex ) && handleTriggerResults->accept( iP->pathIndex );
              triggerObjectStandAlone.addPathName( iP->pathName, pathFired && iP->lastFilter, pathFired && iP->l3Filter );
          }
      }

      triggerObjectsStandAlone->push_back( triggerObjectStandAlone );
      if ( ! onlyStandAlone_ ) {
        triggerObjects->push_back( triggerObject );
        newObjectKeys[ iO ] = trigger::size_type( triggerObjects->size() - 1 );
      }
    }

    // Re-iterate HLT filters and finally produce them in order to account for optionally skipped objects
    if ( ! onlyStandAlone_ ) {
      std::auto_ptr< TriggerFilterCollection > triggerFilters( new TriggerFilterCollection() );
      triggerFilters->reserve( sizeFilters );
      for ( size_t iF = 0; iF < sizeFilters; ++iF ) {
        const std::string nameFilter( handleTriggerEvent->filterTag( iF ).label() );
        const trigger::Keys & keys  = handleTriggerEvent->filterKeys( iF ); // not cached
        const trigger::Vids & types = handleTriggerEvent->filterIds( iF );  // not cached
        TriggerFilter triggerFilter( nameFilter );
        // set filter type
        const std::string typeFilter( hltConfig_.moduleType( nameFilter ) );
        triggerFilter.setType( typeFilter );
        triggerFilter.setSaveTags( hltConfig_.saveTags( nameFilter ) );
        // set keys and trigger object types of used objects
        for ( size_t iK = 0; iK < keys.size(); ++iK ) { // identical to types.size()
          // check, if current object is excluded
          if ( newObjectKeys.find( keys.at( iK ) ) != newObjectKeys.end() ) {
            if ( newObjectKeys[ keys.at( iK ) ] == sizeObjects ) continue;
            triggerFilter.addObjectKey( newObjectKeys[ keys.at( iK ) ] );
            triggerFilter.addTriggerObjectType( types.at( iK ) );
          } else {
            LogWarning( "triggerObjectKey" ) << "TriggerFilter '" << nameFilter << "' requests non-existing TriggerObject key " << keys.at( iK ) << "\n"
                                             << "Skipping object assignment";
          }
        }
        // set status from path info
        std::map< std::string, int >::iterator iS( moduleStates.find( nameFilter ) );
        if ( iS != moduleStates.end() ) {
          if ( ! triggerFilter.setStatus( iS->second ) ) {
            triggerFilter.setStatus( -1 ); // FIXME different code for "unvalid status determined" needed?
          }
        } else {
          triggerFilter.setStatus( -1 ); // FIXME different code for "unknown" needed?
        }
        // store filter
        triggerFilters->push_back( triggerFilter );
      }
      // put HLT filters to event
      iEvent.put( triggerFilters );
    }

    if (packPrescales_) {
        packedPrescales.reset(new PackedTriggerPrescales(handleTriggerResults)); 
        const edm::TriggerNames & names = iEvent.triggerNames(*handleTriggerResults);
        for (unsigned int i = 0, n = names.size(); i < n; ++i) {
            packedPrescales->addPrescaledTrigger(i, hltConfig_.prescaleValue(set, names.triggerName(i)));
        }
        iEvent.put( packedPrescales );
    }

  } // if ( goodHlt )

  // L1 objects
  // (needs to be done after HLT objects, since their x-links with filters rely on their collection keys)

  // map for assignments of objects to conditions
  std::map< L1GtObject, std::vector< unsigned > > l1ObjectTypeMap;
  if ( ! tagL1ExtraMu_.label().empty() ) {
    Handle< l1extra::L1MuonParticleCollection > handleL1ExtraMu;
    iEvent.getByLabel( tagL1ExtraMu_, handleL1ExtraMu );
    if ( handleL1ExtraMu.isValid() ) {
      std::vector< unsigned > muKeys;
      for ( size_t l1Mu = 0; l1Mu < handleL1ExtraMu->size(); ++l1Mu ) {
        if ( mainBxOnly_ && handleL1ExtraMu->at( l1Mu ).bx() != 0 ) continue;
        TriggerObject triggerObject;
        if ( saveL1Refs_ ) {
          const reco::CandidateBaseRef leafCandRef( l1extra::L1MuonParticleRef( handleL1ExtraMu, l1Mu ) );
          triggerObject = TriggerObject( leafCandRef );
        } else {
          const reco::LeafCandidate leafCandidate( *( handleL1ExtraMu->at( l1Mu ).reco::LeafCandidate::clone() ) );
          triggerObject = TriggerObject( leafCandidate );
        }
        triggerObject.setCollection( tagL1ExtraMu_ );
        triggerObject.addTriggerObjectType( trigger::TriggerL1Mu );
        if ( ! onlyStandAlone_ ) triggerObjects->push_back( triggerObject );
        TriggerObjectStandAlone triggerObjectStandAlone( triggerObject );
        triggerObjectsStandAlone->push_back( triggerObjectStandAlone );
        if ( handleL1ExtraMu->at( l1Mu ).bx() == 0 ) muKeys.push_back( triggerObjectsStandAlone->size() - 1 );
      }
      l1ObjectTypeMap.insert( std::make_pair( Mu, muKeys ) );
    } else LogError( "l1ExtraValid" ) << "l1extra::L1MuonParticleCollection product with InputTag '" << tagL1ExtraMu_.encode() << "' not in event";
  }
  if ( ! tagL1ExtraNoIsoEG_.label().empty() ) {
    Handle< l1extra::L1EmParticleCollection > handleL1ExtraNoIsoEG;
    iEvent.getByLabel( tagL1ExtraNoIsoEG_, handleL1ExtraNoIsoEG );
    if ( handleL1ExtraNoIsoEG.isValid() ) {
      std::vector< unsigned > noIsoEGKeys;
      for ( size_t l1NoIsoEG = 0; l1NoIsoEG < handleL1ExtraNoIsoEG->size(); ++l1NoIsoEG ) {
        if ( mainBxOnly_ && handleL1ExtraNoIsoEG->at( l1NoIsoEG ).bx() != 0 ) continue;
        TriggerObject triggerObject;
        if ( saveL1Refs_ ) {
          const reco::CandidateBaseRef leafCandRef( l1extra::L1EmParticleRef( handleL1ExtraNoIsoEG, l1NoIsoEG ) );
          triggerObject = TriggerObject( leafCandRef );
        } else {
          const reco::LeafCandidate leafCandidate( *( handleL1ExtraNoIsoEG->at( l1NoIsoEG ).reco::LeafCandidate::clone() ) );
          triggerObject = TriggerObject( leafCandidate );
        }
        triggerObject.setCollection( tagL1ExtraNoIsoEG_ );
        triggerObject.addTriggerObjectType( trigger::TriggerL1NoIsoEG );
        if ( ! onlyStandAlone_ ) triggerObjects->push_back( triggerObject );
        TriggerObjectStandAlone triggerObjectStandAlone( triggerObject );
        triggerObjectsStandAlone->push_back( triggerObjectStandAlone );
        if ( handleL1ExtraNoIsoEG->at( l1NoIsoEG ).bx() == 0 ) noIsoEGKeys.push_back( triggerObjectsStandAlone->size() - 1 );
      }
      l1ObjectTypeMap.insert( std::make_pair( NoIsoEG, noIsoEGKeys ) );
    } else LogError( "l1ExtraValid" ) << "l1extra::L1EmParticleCollection product with InputTag '" << tagL1ExtraNoIsoEG_.encode() << "' not in event";
  }
  if ( ! tagL1ExtraIsoEG_.label().empty() ) {
    Handle< l1extra::L1EmParticleCollection > handleL1ExtraIsoEG;
    iEvent.getByLabel( tagL1ExtraIsoEG_, handleL1ExtraIsoEG );
    if ( handleL1ExtraIsoEG.isValid() ) {
      std::vector< unsigned > isoEGKeys;
      for ( size_t l1IsoEG = 0; l1IsoEG < handleL1ExtraIsoEG->size(); ++l1IsoEG ) {
        if ( mainBxOnly_ && handleL1ExtraIsoEG->at( l1IsoEG ).bx() != 0 ) continue;
        TriggerObject triggerObject;
        if ( saveL1Refs_ ) {
          const reco::CandidateBaseRef leafCandRef( l1extra::L1EmParticleRef( handleL1ExtraIsoEG, l1IsoEG ) );
          triggerObject = TriggerObject( leafCandRef );
        } else {
          const reco::LeafCandidate leafCandidate( *( handleL1ExtraIsoEG->at( l1IsoEG ).reco::LeafCandidate::clone() ) );
          triggerObject = TriggerObject( leafCandidate );
        }
        triggerObject.setCollection( tagL1ExtraIsoEG_ );
        triggerObject.addTriggerObjectType( trigger::TriggerL1IsoEG );
        if ( ! onlyStandAlone_ ) triggerObjects->push_back( triggerObject );
        TriggerObjectStandAlone triggerObjectStandAlone( triggerObject );
        triggerObjectsStandAlone->push_back( triggerObjectStandAlone );
        if ( handleL1ExtraIsoEG->at( l1IsoEG ).bx() == 0 ) isoEGKeys.push_back( triggerObjectsStandAlone->size() - 1 );
      }
      l1ObjectTypeMap.insert( std::make_pair( IsoEG, isoEGKeys ) );
    } else LogError( "l1ExtraValid" ) << "l1extra::L1EmParticleCollection product with InputTag '" << tagL1ExtraIsoEG_.encode() << "' not in event";
  }
  if ( ! tagL1ExtraCenJet_.label().empty() ) {
    Handle< l1extra::L1JetParticleCollection > handleL1ExtraCenJet;
    iEvent.getByLabel( tagL1ExtraCenJet_, handleL1ExtraCenJet );
    if ( handleL1ExtraCenJet.isValid() ) {
      std::vector< unsigned > cenJetKeys;
      for ( size_t l1CenJet = 0; l1CenJet < handleL1ExtraCenJet->size(); ++l1CenJet ) {
        if ( mainBxOnly_ && handleL1ExtraCenJet->at( l1CenJet ).bx() != 0 ) continue;
        TriggerObject triggerObject;
        if ( saveL1Refs_ ) {
          const reco::CandidateBaseRef leafCandRef( l1extra::L1JetParticleRef( handleL1ExtraCenJet, l1CenJet ) );
          triggerObject = TriggerObject( leafCandRef );
        } else {
          const reco::LeafCandidate leafCandidate( *( handleL1ExtraCenJet->at( l1CenJet ).reco::LeafCandidate::clone() ) );
          triggerObject = TriggerObject( leafCandidate );
        }
        triggerObject.setCollection( tagL1ExtraCenJet_ );
        triggerObject.addTriggerObjectType( trigger::TriggerL1CenJet );
        if ( ! onlyStandAlone_ ) triggerObjects->push_back( triggerObject );
        TriggerObjectStandAlone triggerObjectStandAlone( triggerObject );
        triggerObjectsStandAlone->push_back( triggerObjectStandAlone );
        if ( handleL1ExtraCenJet->at( l1CenJet ).bx() == 0 ) cenJetKeys.push_back( triggerObjectsStandAlone->size() - 1 );
      }
      l1ObjectTypeMap.insert( std::make_pair( CenJet, cenJetKeys ) );
    } else LogError( "l1ExtraValid" ) << "l1extra::L1JetParticleCollection product with InputTag '" << tagL1ExtraCenJet_.encode() << "' not in event";
  }
  if ( ! tagL1ExtraForJet_.label().empty() ) {
    Handle< l1extra::L1JetParticleCollection > handleL1ExtraForJet;
    iEvent.getByLabel( tagL1ExtraForJet_, handleL1ExtraForJet );
    if ( handleL1ExtraForJet.isValid() ) {
      std::vector< unsigned > forJetKeys;
      for ( size_t l1ForJet = 0; l1ForJet < handleL1ExtraForJet->size(); ++l1ForJet ) {
        if ( mainBxOnly_ && handleL1ExtraForJet->at( l1ForJet ).bx() != 0 ) continue;
        TriggerObject triggerObject;
        if ( saveL1Refs_ ) {
          const reco::CandidateBaseRef leafCandRef( l1extra::L1JetParticleRef( handleL1ExtraForJet, l1ForJet ) );
          triggerObject = TriggerObject( leafCandRef );
        } else {
          const reco::LeafCandidate leafCandidate( *( handleL1ExtraForJet->at( l1ForJet ).reco::LeafCandidate::clone() ) );
          triggerObject = TriggerObject( leafCandidate );
        }
        triggerObject.setCollection( tagL1ExtraForJet_ );
        triggerObject.addTriggerObjectType( trigger::TriggerL1ForJet );
        if ( ! onlyStandAlone_ ) triggerObjects->push_back( triggerObject );
        TriggerObjectStandAlone triggerObjectStandAlone( triggerObject );
        triggerObjectsStandAlone->push_back( triggerObjectStandAlone );
        if ( handleL1ExtraForJet->at( l1ForJet ).bx() == 0 ) forJetKeys.push_back( triggerObjectsStandAlone->size() - 1 );
      }
      l1ObjectTypeMap.insert( std::make_pair( ForJet, forJetKeys ) );
    } else LogError( "l1ExtraValid" ) << "l1extra::L1JetParticleCollection product with InputTag '" << tagL1ExtraForJet_.encode() << "' not in event";
  }
  if ( ! tagL1ExtraTauJet_.label().empty() ) {
    Handle< l1extra::L1JetParticleCollection > handleL1ExtraTauJet;
    iEvent.getByLabel( tagL1ExtraTauJet_, handleL1ExtraTauJet );
    if ( handleL1ExtraTauJet.isValid() ) {
      std::vector< unsigned > tauJetKeys;
      for ( size_t l1TauJet = 0; l1TauJet < handleL1ExtraTauJet->size(); ++l1TauJet ) {
        if ( mainBxOnly_ && handleL1ExtraTauJet->at( l1TauJet ).bx() != 0 ) continue;
        TriggerObject triggerObject;
        if ( saveL1Refs_ ) {
          const reco::CandidateBaseRef leafCandRef( l1extra::L1JetParticleRef( handleL1ExtraTauJet, l1TauJet ) );
          triggerObject = TriggerObject( leafCandRef );
        } else {
          const reco::LeafCandidate leafCandidate( *( handleL1ExtraTauJet->at( l1TauJet ).reco::LeafCandidate::clone() ) );
          triggerObject = TriggerObject( leafCandidate );
        }
        triggerObject.setCollection( tagL1ExtraTauJet_ );
        triggerObject.addTriggerObjectType( trigger::TriggerL1TauJet );
        if ( ! onlyStandAlone_ ) triggerObjects->push_back( triggerObject );
        TriggerObjectStandAlone triggerObjectStandAlone( triggerObject );
        triggerObjectsStandAlone->push_back( triggerObjectStandAlone );
        if ( handleL1ExtraTauJet->at( l1TauJet ).bx() == 0 ) tauJetKeys.push_back( triggerObjectsStandAlone->size() - 1 );
      }
      l1ObjectTypeMap.insert( std::make_pair( TauJet, tauJetKeys ) );
    } else LogError( "l1ExtraValid" ) << "l1extra::L1JetParticleCollection product with InputTag '" << tagL1ExtraTauJet_.encode() << "' not in event";
  }
  if ( ! tagL1ExtraETM_ .label().empty()) {
    Handle< l1extra::L1EtMissParticleCollection > handleL1ExtraETM;
    iEvent.getByLabel( tagL1ExtraETM_, handleL1ExtraETM );
    if ( handleL1ExtraETM.isValid() ) {
      std::vector< unsigned > etmKeys;
      for ( size_t l1ETM = 0; l1ETM < handleL1ExtraETM->size(); ++l1ETM ) {
        if ( mainBxOnly_ && handleL1ExtraETM->at( l1ETM ).bx() != 0 ) continue;
        TriggerObject triggerObject;
        if ( saveL1Refs_ ) {
          const reco::CandidateBaseRef leafCandRef( l1extra::L1EtMissParticleRef( handleL1ExtraETM, l1ETM ) );
          triggerObject = TriggerObject( leafCandRef );
        } else {
          const reco::LeafCandidate leafCandidate( *( handleL1ExtraETM->at( l1ETM ).reco::LeafCandidate::clone() ) );
          triggerObject = TriggerObject( leafCandidate );
        }
        triggerObject.setCollection( tagL1ExtraETM_ );
        triggerObject.addTriggerObjectType( trigger::TriggerL1ETM );
        if ( ! onlyStandAlone_ ) triggerObjects->push_back( triggerObject );
        TriggerObjectStandAlone triggerObjectStandAlone( triggerObject );
        triggerObjectsStandAlone->push_back( triggerObjectStandAlone );
        if ( handleL1ExtraETM->at( l1ETM ).bx() == 0 ) etmKeys.push_back( triggerObjectsStandAlone->size() - 1 );
      }
      l1ObjectTypeMap.insert( std::make_pair( ETM, etmKeys ) );
    } else LogError( "l1ExtraValid" ) << "l1extra::L1EtMissParticleCollection product with InputTag '" << tagL1ExtraETM_.encode() << "' not in event";
  }
  if ( ! tagL1ExtraHTM_.label().empty() ) {
    Handle< l1extra::L1EtMissParticleCollection > handleL1ExtraHTM;
    iEvent.getByLabel( tagL1ExtraHTM_, handleL1ExtraHTM );
    if ( handleL1ExtraHTM.isValid() ) {
      std::vector< unsigned > htmKeys;
      for ( size_t l1HTM = 0; l1HTM < handleL1ExtraHTM->size(); ++l1HTM ) {
        if ( mainBxOnly_ && handleL1ExtraHTM->at( l1HTM ).bx() != 0 ) continue;
        TriggerObject triggerObject;
        if ( saveL1Refs_ ) {
          const reco::CandidateBaseRef leafCandRef( l1extra::L1EtMissParticleRef( handleL1ExtraHTM, l1HTM ) );
          triggerObject = TriggerObject( leafCandRef );
        } else {
          const reco::LeafCandidate leafCandidate( *( handleL1ExtraHTM->at( l1HTM ).reco::LeafCandidate::clone() ) );
          triggerObject = TriggerObject( leafCandidate );
        }
        triggerObject.setCollection( tagL1ExtraHTM_ );
        triggerObject.addTriggerObjectType( trigger::TriggerL1HTM );
        if ( ! onlyStandAlone_ ) triggerObjects->push_back( triggerObject );
        TriggerObjectStandAlone triggerObjectStandAlone( triggerObject );
        triggerObjectsStandAlone->push_back( triggerObjectStandAlone );
        if ( handleL1ExtraHTM->at( l1HTM ).bx() == 0 ) htmKeys.push_back( triggerObjectsStandAlone->size() - 1 );
      }
      l1ObjectTypeMap.insert( std::make_pair( HTM, htmKeys ) );
    } else LogError( "l1ExtraValid" ) << "l1extra::L1EtMissParticleCollection product with InputTag '" << tagL1ExtraHTM_.encode() << "' not in event";
  }

  // Put trigger objects to event
  if ( ! onlyStandAlone_ ) iEvent.put( triggerObjects );

  // L1 algorithms
  if ( ! onlyStandAlone_ ) {
    std::auto_ptr< TriggerAlgorithmCollection > triggerAlgos( new TriggerAlgorithmCollection() );
    std::auto_ptr< TriggerConditionCollection > triggerConditions( new TriggerConditionCollection() );
    if ( addL1Algos_ ) {
      // create trigger object types transalation map (yes, it's ugly!)
      std::map< L1GtObject, trigger::TriggerObjectType > mapObjectTypes;
      mapObjectTypes.insert( std::make_pair( Mu     , trigger::TriggerL1Mu ) );
      mapObjectTypes.insert( std::make_pair( NoIsoEG, trigger::TriggerL1NoIsoEG ) );
      mapObjectTypes.insert( std::make_pair( IsoEG  , trigger::TriggerL1IsoEG ) );
      mapObjectTypes.insert( std::make_pair( CenJet , trigger::TriggerL1CenJet ) );
      mapObjectTypes.insert( std::make_pair( ForJet , trigger::TriggerL1ForJet ) );
      mapObjectTypes.insert( std::make_pair( TauJet , trigger::TriggerL1TauJet ) );
      mapObjectTypes.insert( std::make_pair( ETM    , trigger::TriggerL1ETM ) );
      mapObjectTypes.insert( std::make_pair( HTM    , trigger::TriggerL1HTM ) );
      // get and cache L1 menu
      l1GtUtils_.getL1GtRunCache( iEvent, iSetup, useL1EventSetup, useL1GtTriggerMenuLite );
      ESHandle< L1GtTriggerMenu > handleL1GtTriggerMenu;
      iSetup.get< L1GtTriggerMenuRcd >().get( handleL1GtTriggerMenu );
      L1GtTriggerMenu l1GtTriggerMenu( *handleL1GtTriggerMenu );
      const AlgorithmMap l1GtAlgorithms( l1GtTriggerMenu.gtAlgorithmMap() );
      const AlgorithmMap l1GtTechTriggers( l1GtTriggerMenu.gtTechnicalTriggerMap() );
      l1GtTriggerMenu.buildGtConditionMap();
      const std::vector< ConditionMap > l1GtConditionsVector( l1GtTriggerMenu.gtConditionMap() );
      // cache conditions in one single condition map
      ConditionMap l1GtConditions;
      for ( size_t iCv = 0; iCv < l1GtConditionsVector.size(); ++iCv ) {
        l1GtConditions.insert( l1GtConditionsVector.at( iCv ).begin(), l1GtConditionsVector.at( iCv ).end() );
      }
      triggerAlgos->reserve( l1GtAlgorithms.size() + l1GtTechTriggers.size() );
      Handle< L1GlobalTriggerObjectMaps > handleL1GlobalTriggerObjectMaps;
      iEvent.getByToken( l1GlobalTriggerObjectMapsToken_, handleL1GlobalTriggerObjectMaps );
      if( ! handleL1GlobalTriggerObjectMaps.isValid() ) {
        LogError( "l1ObjectMap" ) << "L1GlobalTriggerObjectMaps product with InputTag '" << tagL1GlobalTriggerObjectMaps_.encode() << "' not in event\n"
                                    << "No L1 objects and GTL results available for physics algorithms";
      }
      handleL1GlobalTriggerObjectMaps->consistencyCheck();
      if ( firstInRun_ ) {
        l1PSet_ = ( ParameterSet* )( pset::Registry::instance()->getMapped(handleL1GlobalTriggerObjectMaps->namesParameterSetID()) );
        if (l1PSet_ == 0) {
          LogError( "l1ObjectMap" ) << "ParameterSet registry not available\n"
                                    << "Skipping conditions for all L1 physics algorithm names in this run";
        }
      } else {
        if (l1PSet_ == 0) {
          LogInfo( "l1ObjectMap" ) << "ParameterSet registry not available\n"
                                   << "Skipping conditions for all L1 physics algorithm names in this event";
        }
      }
      // physics algorithms
      for ( CItAlgo iAlgo = l1GtAlgorithms.begin(); iAlgo != l1GtAlgorithms.end(); ++iAlgo ) {
        const std::string & algoName( iAlgo->second.algoName() );
        if ( ! ( iAlgo->second.algoBitNumber() < int( L1GlobalTriggerReadoutSetup::NumberPhysTriggers ) ) ) {
          LogError( "l1Algo" ) << "L1 physics algorithm '" << algoName << "' has bit number " << iAlgo->second.algoBitNumber() << " >= " << L1GlobalTriggerReadoutSetup::NumberPhysTriggers << "\n"
                               << "Skipping";
          continue;
        }
        L1GtUtils::TriggerCategory category;
        int bit;
        if ( ! l1GtUtils_.l1AlgoTechTrigBitNumber( algoName, category, bit ) ) {
          LogError( "l1Algo" ) << "L1 physics algorithm '" << algoName << "' not found in the L1 menu\n"
                               << "Skipping";
          continue;
        }
        if ( category != L1GtUtils::AlgorithmTrigger ) {
          LogError( "l1Algo" ) << "L1 physics algorithm '" << algoName << "' does not have category 'AlgorithmTrigger' from 'L1GtUtils'\n"
                               << "Skipping";
          continue;
        }
        bool decisionBeforeMask;
        bool decisionAfterMask;
        int  prescale;
        int  mask;
        int  error( l1GtUtils_.l1Results( iEvent, algoName, decisionBeforeMask, decisionAfterMask, prescale, mask ) );
        if ( error ) {
          LogError( "l1Algo" ) << "L1 physics algorithm '" << algoName << "' decision has error code " << error << " from 'L1GtUtils'\n"
                               << "Skipping";
          continue;
        }
        TriggerAlgorithm triggerAlgo( algoName, iAlgo->second.algoAlias(), category == L1GtUtils::TechnicalTrigger, (unsigned)bit, (unsigned)prescale, (bool)mask, decisionBeforeMask, decisionAfterMask );
        triggerAlgo.setLogicalExpression( iAlgo->second.algoLogicalExpression() );
        // GTL result and used conditions in physics algorithm
        if( ! handleL1GlobalTriggerObjectMaps.isValid() ) {
          triggerAlgos->push_back( triggerAlgo );
          continue; // LogWarning already earlier (before loop)
        }
        if ( ! handleL1GlobalTriggerObjectMaps->algorithmExists(bit)) {
          LogError( "l1ObjectMap" ) << "L1 physics algorithm '" << algoName << "' is missing in L1GlobalTriggerObjectMaps\n"
                                    << "Skipping conditions and GTL result";
          triggerAlgos->push_back( triggerAlgo );
          continue;
        }
        bool algorithmResult = handleL1GlobalTriggerObjectMaps->algorithmResult(bit);
//         if ( ( algorithmResult != decisionBeforeMask ) && ( decisionBeforeMask == true || prescale == 1 ) ) {
        if ( ( algorithmResult != decisionBeforeMask ) && ( decisionBeforeMask == true ) ) { // FIXME: understand the difference for un-prescaled algos 118, 119, 123
          LogInfo( "l1ObjectMap" ) << "L1 physics algorithm '" << algoName << "' with different decisions in\n"
                                   << "L1GlobalTriggerObjectMaps (GTL result)        : " << algorithmResult << "\n"
                                   << "L1GlobalTriggerReadoutRecord (decision before mask): " << decisionBeforeMask;
        }
        triggerAlgo.setGtlResult( algorithmResult );
        // conditions in algorithm
        L1GlobalTriggerObjectMaps::ConditionsInAlgorithm conditions = handleL1GlobalTriggerObjectMaps->getConditionsInAlgorithm(bit);
        if (l1PSet_ == 0) {
          triggerAlgos->push_back( triggerAlgo );
          continue;
        }
        if (!l1PSet_->exists(algoName)) {
          if ( firstInRun_ ) {
            LogError( "l1ObjectMap" ) << "L1 physics algorithm name '" << algoName << "' not available in ParameterSet registry\n"
                                      << "Skipping conditions for this algorithm in this run";
          } else {
            LogInfo( "l1ObjectMap" ) << "L1 physics algorithm name '" << algoName << "' not available in ParameterSet registry\n"
                                     << "Skipping conditions for this algorithm in this event";
          }
          triggerAlgos->push_back( triggerAlgo );
          continue;
        }
        std::vector<std::string> conditionNames( l1PSet_->getParameter<std::vector<std::string> >(algoName) );

        for ( unsigned iT = 0; iT < conditionNames.size(); ++iT ) {
          size_t key( triggerConditions->size() );
          for ( size_t iC = 0; iC < triggerConditions->size(); ++iC ) {
            if ( conditionNames.at(iT) == triggerConditions->at( iC ).name() ) {
              key = iC;
              break;
            }
          }
          if ( key == triggerConditions->size() ) {
            if ( iT >= conditions.nConditions() ) {
              LogError( "l1CondMap" ) << "More condition names from ParameterSet registry than the " << conditions.nConditions() << " conditions in L1GlobalTriggerObjectMaps\n"
                                      << "Skipping condition " << conditionNames.at(iT) << " in algorithm " << algoName;
              break;
            }
            TriggerCondition triggerCond( conditionNames[iT], conditions.getConditionResult(iT) );
            if ( l1GtConditions.find( triggerCond.name() ) != l1GtConditions.end() ) {
              triggerCond.setCategory( l1GtConditions[ triggerCond.name() ]->condCategory() );
              triggerCond.setType( l1GtConditions[ triggerCond.name() ]->condType() );
              const std::vector< L1GtObject > l1ObjectTypes( l1GtConditions[ triggerCond.name() ]->objectType() );
              for ( size_t iType = 0 ; iType < l1ObjectTypes.size(); ++iType ) {
                triggerCond.addTriggerObjectType( mapObjectTypes[ l1ObjectTypes.at( iType ) ] );
              }
              // objects in condition
              L1GlobalTriggerObjectMaps::CombinationsInCondition combinations = handleL1GlobalTriggerObjectMaps->getCombinationsInCondition(bit, iT);
              for ( size_t iVV = 0; iVV < combinations.nCombinations(); ++iVV ) {
                for ( size_t iV = 0; iV < combinations.nObjectsPerCombination(); ++iV ) {

                  unsigned objectIndex = combinations.getObjectIndex(iVV, iV);
                  if ( iV >= l1ObjectTypes.size() ) {
                    LogError( "l1CondMap" ) << "Index " << iV << " in combinations vector overshoots size " << l1ObjectTypes.size() << " of types vector in conditions map\n"
                                            << "Skipping object key in condition " << triggerCond.name();
                  } else if ( l1ObjectTypeMap.find( l1ObjectTypes.at( iV ) ) != l1ObjectTypeMap.end() ) {
                    if ( objectIndex >= l1ObjectTypeMap[ l1ObjectTypes.at( iV ) ].size() ) {
                      LogError( "l1CondMap" ) << "Index " << objectIndex << " in combination overshoots number " << l1ObjectTypeMap[ l1ObjectTypes.at( iV ) ].size() << "of according trigger objects\n"
                                              << "Skipping object key in condition " << triggerCond.name();
                    }
                    const unsigned objectKey( l1ObjectTypeMap[ l1ObjectTypes.at( iV ) ].at( objectIndex ) );
                    triggerCond.addObjectKey( objectKey );
                    // add current condition and algorithm also to the according stand-alone trigger object
                    triggerObjectsStandAlone->at( objectKey ).addAlgorithmName( triggerAlgo.name(), ( triggerAlgo.decision() && triggerCond.wasAccept() ) );
                    triggerObjectsStandAlone->at( objectKey ).addConditionName( triggerCond.name() );
                  }
                }
              }
            } else {
              LogWarning( "l1CondMap" ) << "L1 conditions '" << triggerCond.name() << "' not found in the L1 menu\n"
                                        << "Remains incomplete";
            }
            triggerConditions->push_back( triggerCond );
          }
          triggerAlgo.addConditionKey( key );
        }
        triggerAlgos->push_back( triggerAlgo );
      }
      // technical triggers
      for ( CItAlgo iAlgo = l1GtTechTriggers.begin(); iAlgo != l1GtTechTriggers.end(); ++iAlgo ) {
        const std::string & algoName( iAlgo->second.algoName() );
        if ( ! ( iAlgo->second.algoBitNumber() < int( L1GlobalTriggerReadoutSetup::NumberTechnicalTriggers ) ) ) {
          LogError( "l1Algo" ) << "L1 technical trigger '" << algoName << "' has bit number " << iAlgo->second.algoBitNumber() << " >= " << L1GlobalTriggerReadoutSetup::NumberTechnicalTriggers << "\n"
                               << "Skipping";
          continue;
        }
        L1GtUtils::TriggerCategory category;
        int bit;
        if ( ! l1GtUtils_.l1AlgoTechTrigBitNumber( algoName, category, bit ) ) {
          LogError( "l1Algo" ) << "L1 technical trigger '" << algoName << "' not found in the L1 menu\n"
                               << "Skipping";
          continue;
        }
        if ( category != L1GtUtils::TechnicalTrigger ) {
          LogError( "l1Algo" ) << "L1 technical trigger '" << algoName << "' does not have category 'TechnicalTrigger' from 'L1GtUtils'\n"
                               << "Skipping";
          continue;
        }
        bool decisionBeforeMask;
        bool decisionAfterMask;
        int  prescale;
        int  mask;
        int  error( l1GtUtils_.l1Results( iEvent, algoName, decisionBeforeMask, decisionAfterMask, prescale, mask ) );
        if ( error ) {
          LogError( "l1Algo" ) << "L1 technical trigger '" << algoName << "' decision has error code " << error << " from 'L1GtUtils'\n"
                               << "Skipping";
          continue;
        }
        TriggerAlgorithm triggerAlgo( algoName, iAlgo->second.algoAlias(), category == L1GtUtils::TechnicalTrigger, (unsigned)bit, (unsigned)prescale, (bool)mask, decisionBeforeMask, decisionAfterMask );
        triggerAlgo.setLogicalExpression( iAlgo->second.algoLogicalExpression() );
        triggerAlgos->push_back( triggerAlgo );
      }
    }

    // Put L1 algorithms and conditions to event
    iEvent.put( triggerAlgos );
    iEvent.put( triggerConditions );
  }


  if (packPathNames_) {
    const edm::TriggerNames & names = iEvent.triggerNames(*handleTriggerResults);
    for (pat::TriggerObjectStandAlone &obj : *triggerObjectsStandAlone) {
      obj.packPathNames(names);
    }
  }
  // Put (finally) stand-alone trigger objects to event
  iEvent.put( triggerObjectsStandAlone );

  firstInRun_ = false;

}

void PATTriggerProducer::ModuleLabelToPathAndFlags::init(const HLTConfigProvider &hltConfig_) {
    clear();
    const std::vector<std::string> & pathNames = hltConfig_.triggerNames();
    unsigned int sizePaths = pathNames.size();
    for ( unsigned int indexPath = 0; indexPath < sizePaths; ++indexPath ) {
        const std::string & namePath = pathNames[indexPath];

        const std::vector<std::string> & nameModules = hltConfig_.moduleLabels(indexPath);
        unsigned int sizeModulesPath = nameModules.size();
        bool lastFilter = true;
        unsigned int iM = sizeModulesPath;
        while (iM > 0) {
            const std::string & nameFilter = nameModules[--iM];
            if (hltConfig_.moduleEDMType(nameFilter) != "EDFilter") continue;
            if (hltConfig_.moduleType(nameFilter)    == "HLTBool")  continue;
            bool saveTags = hltConfig_.saveTags(nameFilter);
            insert( nameFilter, namePath, indexPath, lastFilter, saveTags );
            if (saveTags) lastFilter = false; // FIXME: rather always?
        }
    }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PATTriggerProducer );
