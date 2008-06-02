//
// $Id: PATHLTProducer.cc,v 1.1.2.5 2008/04/24 18:51:33 vadler Exp $
//


#include "PhysicsTools/PatAlgos/plugins/PATHLTProducer.h"

#include <vector>

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"

#include "DataFormats/PatCandidates/interface/TriggerPrimitive.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"


using namespace pat;
using namespace edm;
using namespace std;


PATHLTProducer::PATHLTProducer( const ParameterSet& iConfig ) :
  // initialize
  triggerResults_ (iConfig.getParameter<InputTag>( "triggerResults" ) ),
  triggerName_    (iConfig.getParameter<string>  ( "triggerName" ) ),
  filterName_     (iConfig.getParameter<InputTag>( "filterName" ) )
{
  produces<TriggerPrimitiveCollection>();
}


PATHLTProducer::~PATHLTProducer()
{
}


void PATHLTProducer::produce( Event& iEvent, const EventSetup& iSetup )
{
  auto_ptr<TriggerPrimitiveCollection> patHltCandidates( new TriggerPrimitiveCollection );
  edm::Handle<TriggerResults> triggerResults;
  iEvent.getByLabel( triggerResults_, triggerResults );
  TriggerNames triggerNames( *triggerResults );
  unsigned int triggerIndex = triggerNames.triggerIndex( triggerName_ );
  if ( triggerIndex == triggerNames.size() ) {
    LogDebug( "wrongTriggerPath" ) << "PATHLTProducer: The trigger path " << triggerName_ << " is not known in this event!";
  } else if ( ! triggerResults->wasrun( triggerIndex ) ) {
    LogDebug( "notrunTriggerPath" ) << "PATHLTProducer: The trigger path " << triggerName_ << " was not run in this event!";
  } else if ( ! triggerResults->accept( triggerIndex ) ) {
    LogDebug( "notacceptTriggerPath" ) << "PATHLTProducer: The trigger path " << triggerName_ << " did not accept this event!";
  } else if (   triggerResults->error ( triggerIndex ) ) {
    LogDebug( "errorTriggerPath" ) << "PATHLTProducer: The trigger path " << triggerName_ << " had an error in this event!";
  } else {
    Handle<reco::HLTFilterObjectWithRefs> hltFilter;
    try { // In this case, we want to act differently compared to the usual behaviour on "ProductNotFound" exception thrown by Event::getByLabel.
      iEvent.getByLabel( filterName_, hltFilter );
      if ( triggerIndex != hltFilter->path() ) {
        LogDebug( "wrongTriggerModule" ) << "PATHLTProducer: The filter module " << filterName_.label() << " does not belong to the trigger path " << triggerName_ << "!";
      } else {
        // loop over trigger objects and store trigger candidates
        for ( unsigned int iTriggeredObject = 0; iTriggeredObject < hltFilter->size(); ++iTriggeredObject ) {
          auto_ptr<TriggerPrimitive> ptr( new TriggerPrimitive( (hltFilter->at(iTriggeredObject)).p4(), triggerName_, filterName_.label() ) );
          patHltCandidates->push_back( ptr );        
        }  
      }
    } catch( Exception exc ) {
      if ( exc.categoryCode() == errors::ProductNotFound ) {
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
