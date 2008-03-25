//
// $Id: PATHLTProducer.cc,v 1.1 2008/02/26 13:59:12 vadler Exp $
//


#include "PhysicsTools/PatAlgos/plugins/PATHLTProducer.h"

#include <vector>

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace pat;

PATHLTProducer::PATHLTProducer( const ParameterSet& iConfig ) :
  // initialize
  triggerResults_ (iConfig.getParameter<InputTag>( "triggerResults" ) ),
  triggerName_    (iConfig.getParameter<string>  ( "triggerName" ) ),
  filterName_     (iConfig.getParameter<InputTag>( "filterName" ) )
{
  produces<reco::CandidateCollection>();
}


PATHLTProducer::~PATHLTProducer()
{
}


void PATHLTProducer::produce( Event& iEvent, const EventSetup& iSetup )
{
  auto_ptr<reco::CandidateCollection> patHltCandidates( new reco::CandidateCollection );
  Handle<TriggerResults> triggerResults;
  iEvent.getByLabel( triggerResults_, triggerResults );
  TriggerNames triggerNames( *triggerResults );
  unsigned int triggerIndex = triggerNames.triggerIndex( triggerName_ );
  if ( triggerIndex == triggerNames.size() ) {
    LogWarning( "wrongTriggerPath" ) << "PATHLTProducer: The trigger path " << triggerName_ << " is not known in this event!";
  } else if ( ! triggerResults->wasrun( triggerIndex ) ) {
    LogWarning( "notrunTriggerPath" ) << "PATHLTProducer: The trigger path " << triggerName_ << " was not run in this event!";
  } else if ( ! triggerResults->accept( triggerIndex ) ) {
    LogWarning( "notacceptTriggerPath" ) << "PATHLTProducer: The trigger path " << triggerName_ << " did not accept this event!";
  } else if (   triggerResults->error ( triggerIndex ) ) {
    LogWarning( "errorTriggerPath" ) << "PATHLTProducer: The trigger path " << triggerName_ << " had an error in this event!";
  } else {
    Handle<reco::HLTFilterObjectWithRefs> hltFilter;
    try { // In this case, we want to act differently compared to the usual behaviour on "ProductNotFound" exception thrown by Event::getByLabel.
      iEvent.getByLabel( filterName_, hltFilter );
      if ( triggerIndex != hltFilter->path() ) {
        LogWarning( "wrongTriggerModule" ) << "PATHLTProducer: The filter module " << filterName_.label() << " does not belong to the trigger path " << triggerName_ << "!";
      } else {
        // loop over trigger objects and store trigger candidates
        for ( unsigned int iTriggerObject = 0; iTriggerObject < hltFilter->size(); ++iTriggerObject ) {
          const reco::Candidate * patHltCandidate( &(hltFilter->at(iTriggerObject)) );
          auto_ptr<reco::Candidate> ptr( patHltCandidate->clone() );
          patHltCandidates->push_back( ptr );        
        }  
      }
    } catch( Exception exc ) {
      if ( exc.codeToString( exc.categoryCode() ) == "ProductNotFound" ) {
        LogWarning( "notpresentTriggerModule" ) << "PATHLTProducer: The filter module " << filterName_.label() << " is not present here!";
      } else {
        throw exc;
      }
    }
  }
  iEvent.put( patHltCandidates );
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATHLTProducer);
