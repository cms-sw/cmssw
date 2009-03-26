//
// $Id: PATTriggerEventProducer.cc,v 1.1.2.3 2009/03/18 19:08:22 vadler Exp $
//


#include "PhysicsTools/PatAlgos/plugins/PATTriggerEventProducer.h"

// #include "DataFormats/Common/interface/AssociativeIterator.h"


using namespace pat;

PATTriggerEventProducer::PATTriggerEventProducer( const edm::ParameterSet & iConfig ) :
  nameProcess_( iConfig.getParameter< std::string >( "processName" ) ),
  tagTriggerResults_( iConfig.getParameter< edm::InputTag >( "triggerResults" ) ),
  tagTriggerEvent_( iConfig.getParameter< edm::InputTag >( "triggerEvent" ) ),
  tagTriggerProducer_( iConfig.getParameter< edm::InputTag >( "patTriggerProducer" ) ),
  tagsTriggerMatcher_( iConfig.getParameter< std::vector< edm::InputTag > >( "patTriggerMatches" ) )
{
  if ( tagTriggerResults_.process().empty() ) {
    tagTriggerResults_ = edm::InputTag( tagTriggerResults_.label(), tagTriggerResults_.instance(), nameProcess_ );
  }
  if ( tagTriggerEvent_.process().empty() ) {
    tagTriggerEvent_ = edm::InputTag( tagTriggerEvent_.label(), tagTriggerEvent_.instance(), nameProcess_ );
  }

  produces< TriggerEvent >();
}

PATTriggerEventProducer::~PATTriggerEventProducer()
{
}

void PATTriggerEventProducer::beginRun( edm::Run & iRun, const edm::EventSetup & iSetup )
{
  if ( ! hltConfig_.init( nameProcess_ ) ) {
    edm::LogError( "hltConfigExtraction" ) << "HLT config extraction err with process name " << nameProcess_;
    return;
  }                          
}

void PATTriggerEventProducer::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  if ( hltConfig_.size() <= 0 ) {
    edm::LogError( "hltConfigSize" ) << "HLT config size err" << "\n"
                                     << "Check for occurence of an \"errHltConfigExtraction\" from beginRun()";
    return;
  }
  edm::Handle< edm::TriggerResults > handleTriggerResults;
  iEvent.getByLabel( tagTriggerResults_, handleTriggerResults );
  if ( ! handleTriggerResults.isValid() ) {
    edm::LogError( "triggerResultsValid" ) << "edm::TriggerResults product with InputTag " << tagTriggerResults_.encode() << " not in event";
    return;
  }
  edm::Handle< trigger::TriggerEvent > handleTriggerEvent;
  iEvent.getByLabel( tagTriggerEvent_, handleTriggerEvent );
  if ( ! handleTriggerEvent.isValid() ) {
    edm::LogError( "triggerEventValid" ) << "trigger::TriggerEvent product with InputTag " << tagTriggerEvent_.encode() << " not in event";
    return;
  }
  edm::Handle< TriggerPathCollection > handleTriggerPaths;
  iEvent.getByLabel( tagTriggerProducer_, handleTriggerPaths );
  edm::Handle< TriggerFilterCollection > handleTriggerFilters;
  iEvent.getByLabel( tagTriggerProducer_, handleTriggerFilters );
  edm::Handle< TriggerObjectCollection > handleTriggerObjects;
  iEvent.getByLabel( tagTriggerProducer_, handleTriggerObjects );

  // produce trigger event
  
  std::auto_ptr< TriggerEvent > triggerEvent( new TriggerEvent( std::string( hltConfig_.tableName() ), handleTriggerResults->wasrun(), handleTriggerResults->accept(), handleTriggerResults->error() ) );
  // set product references to trigger collections
  if ( handleTriggerPaths.isValid() ) {
    triggerEvent->setPaths( handleTriggerPaths );
  } else {
    edm::LogError( "triggerPathsValid" ) << "pat::TriggerPathCollection product with InputTag " << tagTriggerProducer_.encode() << " not in event";
  }
  if ( handleTriggerFilters.isValid() ) {
    triggerEvent->setFilters( handleTriggerFilters );
  } else {
    edm::LogError( "triggerFiltersValid" ) << "pat::TriggerFilterCollection product with InputTag " << tagTriggerProducer_.encode() << " not in event";
  }
  if ( handleTriggerObjects.isValid() ) {
    triggerEvent->setObjects( handleTriggerObjects );
  } else {
    edm::LogError( "triggerObjectsValid" ) << "pat::TriggerObjectCollection product with InputTag " << tagTriggerProducer_.encode() << " not in event";
  }
  // set product references to trigger match association
  for ( size_t iMatch = 0; iMatch < tagsTriggerMatcher_.size(); ++iMatch ) {
    const std::string labelTriggerObjectMatcher( tagsTriggerMatcher_.at( iMatch ).encode() );
    edm::Handle< TriggerObjectMatch > handleTriggerMatch;
    iEvent.getByLabel( labelTriggerObjectMatcher, handleTriggerMatch );
    if ( ! handleTriggerMatch.isValid() ) {
      edm::LogError( "triggerMatchValid" ) << "pat::TriggerObjectMatch product with InputTag " << labelTriggerObjectMatcher << " not in event";
      continue;
    }
    if ( ! ( triggerEvent->addObjectMatchResult( handleTriggerMatch, labelTriggerObjectMatcher ) ) ) {
      edm::LogWarning( "triggerObjectMatchReplication" ) << "pat::TriggerEvent contains already a pat::TriggerObjectMatch from matcher module " << labelTriggerObjectMatcher;
    }
  }
  
  iEvent.put( triggerEvent );
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PATTriggerEventProducer );
