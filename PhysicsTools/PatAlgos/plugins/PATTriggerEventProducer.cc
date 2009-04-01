//
// $Id: PATTriggerEventProducer.cc,v 1.1.2.4 2009/03/27 21:31:06 vadler Exp $
//


#include "PhysicsTools/PatAlgos/plugins/PATTriggerEventProducer.h"

#include <cassert>

#include "DataFormats/Common/interface/AssociativeIterator.h"


using namespace pat;

PATTriggerEventProducer::PATTriggerEventProducer( const edm::ParameterSet & iConfig ) :
  nameProcess_( iConfig.getParameter< std::string >( "processName" ) ),
  tagTriggerResults_( iConfig.getParameter< edm::InputTag >( "triggerResults" ) ),
  tagTriggerProducer_( iConfig.getParameter< edm::InputTag >( "patTriggerProducer" ) ),
  tagsTriggerMatcher_( iConfig.getParameter< std::vector< edm::InputTag > >( "patTriggerMatches" ) )
{
  if ( tagTriggerResults_.process().empty() ) {
    tagTriggerResults_ = edm::InputTag( tagTriggerResults_.label(), tagTriggerResults_.instance(), nameProcess_ );
  }

  for ( size_t iMatch = 0; iMatch < tagsTriggerMatcher_.size(); ++iMatch ) {
   produces< TriggerObjectMatch >( tagsTriggerMatcher_.at( iMatch ).label() );
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
  edm::Handle< TriggerPathCollection > handleTriggerPaths;
  iEvent.getByLabel( tagTriggerProducer_, handleTriggerPaths );
  edm::Handle< TriggerFilterCollection > handleTriggerFilters;
  iEvent.getByLabel( tagTriggerProducer_, handleTriggerFilters );
  edm::Handle< TriggerObjectCollection > handleTriggerObjects;
  iEvent.getByLabel( tagTriggerProducer_, handleTriggerObjects );
  edm::Handle< TriggerObjectStandAloneCollection > handleTriggerObjectsStandAlone;
  iEvent.getByLabel( tagTriggerProducer_, handleTriggerObjectsStandAlone );
  assert( handleTriggerObjects->size() == handleTriggerObjectsStandAlone->size() );

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
  
  // produce trigger match association and set references
  if ( handleTriggerObjects.isValid() ) {
    for ( size_t iMatch = 0; iMatch < tagsTriggerMatcher_.size(); ++iMatch ) {
      const std::string labelTriggerObjectMatcher( tagsTriggerMatcher_.at( iMatch ).label() );
      // copy trigger match association using TriggerObjectStandAlone to those using TriggerObject
      // relying on the fact, that only one candidate collection is present in the association
      edm::Handle< TriggerObjectStandAloneMatch > handleTriggerObjectStandAloneMatch;
      iEvent.getByLabel( labelTriggerObjectMatcher, handleTriggerObjectStandAloneMatch );
      if ( ! handleTriggerObjectStandAloneMatch.isValid() ) {
        edm::LogError( "triggerMatchValid" ) << "pat::TriggerObjectStandAloneMatch product with InputTag " << labelTriggerObjectMatcher << " not in event";
        continue;
      }
      edm::AssociativeIterator< reco::CandidateBaseRef, TriggerObjectStandAloneMatch > it( *handleTriggerObjectStandAloneMatch, edm::EdmEventItemGetter< reco::CandidateBaseRef >( iEvent ) ), itEnd( it.end() );
      edm::Handle< reco::CandidateView > handleCands;
      std::vector< int > indices;
      while ( it != itEnd ) {
        if ( indices.size() == 0 ) {
          iEvent.get( it->first.id(), handleCands );
        }
        indices.push_back( it->second.key() );
        ++it;
      }
      std::auto_ptr< TriggerObjectMatch > triggerObjectMatch( new TriggerObjectMatch( handleTriggerObjects ) );
      TriggerObjectMatch::Filler matchFiller( *triggerObjectMatch );
      if ( handleCands.isValid() ) {
        matchFiller.insert( handleCands, indices.begin(), indices.end() );
      }
      matchFiller.fill();
      edm::OrphanHandle< TriggerObjectMatch > handleTriggerObjectMatch( iEvent.put( triggerObjectMatch, labelTriggerObjectMatcher ) );
      // set product reference to trigger match association
      if ( ! handleTriggerObjectMatch.isValid() ) {
        edm::LogError( "triggerMatchValid" ) << "pat::TriggerObjectMatch product with InputTag " << labelTriggerObjectMatcher << " not in event";
        continue;
      }
      if ( ! ( triggerEvent->addObjectMatchResult( handleTriggerObjectMatch, labelTriggerObjectMatcher ) ) ) {
        edm::LogWarning( "triggerObjectMatchReplication" ) << "pat::TriggerEvent contains already a pat::TriggerObjectMatch from matcher module " << labelTriggerObjectMatcher;
      }
    }
  }
  
  iEvent.put( triggerEvent );
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PATTriggerEventProducer );
