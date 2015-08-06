// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      pat::PATTriggerObjectStandAloneUnpacker
//
//
/**
  \class    pat::PATTriggerObjectStandAloneUnpacker PATTriggerObjectStandAloneUnpacker.h "PhysicsTools/PatAlgos/plugins/PATTriggerObjectStandAloneUnpacker.cc"
  \brief    Unpacks a pat::TriggerObjectStandAloneCollection with packed path names.

  The producer will throw, if a pat::TriggerObjectStandAloneCollection with unpacked path names is used as input.

  \author   Volker Adler
*/


#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/Common/interface/TriggerResults.h"

namespace pat {
  
  class PATTriggerObjectStandAloneUnpacker : public edm::global::EDProducer<> {
    
  public:
    
    explicit PATTriggerObjectStandAloneUnpacker( const edm::ParameterSet & iConfig );
    ~PATTriggerObjectStandAloneUnpacker() {};
    
  private:
    
    virtual void produce(edm::StreamID, edm::Event & iEvent, const edm::EventSetup& iSetup) const override;
    
    const edm::EDGetTokenT< TriggerObjectStandAloneCollection > patTriggerObjectsStandAloneToken_;
    const edm::EDGetTokenT< edm::TriggerResults > triggerResultsToken_;
    
  };
  
}


using namespace pat;


PATTriggerObjectStandAloneUnpacker::PATTriggerObjectStandAloneUnpacker( const edm::ParameterSet & iConfig )
: patTriggerObjectsStandAloneToken_( consumes< TriggerObjectStandAloneCollection >( iConfig.getParameter< edm::InputTag >( "patTriggerObjectsStandAlone" ) ) )
, triggerResultsToken_( consumes< edm::TriggerResults >( iConfig.getParameter< edm::InputTag >( "triggerResults" ) ) )
{
  produces< TriggerObjectStandAloneCollection >();
}

void PATTriggerObjectStandAloneUnpacker::produce( edm::StreamID, edm::Event & iEvent, const edm::EventSetup& iSetup) const
{
  edm::Handle< TriggerObjectStandAloneCollection > patTriggerObjectsStandAlone;
  iEvent.getByToken( patTriggerObjectsStandAloneToken_, patTriggerObjectsStandAlone );
  edm::Handle< edm::TriggerResults > triggerResults;
  iEvent.getByToken( triggerResultsToken_, triggerResults );

  std::auto_ptr< TriggerObjectStandAloneCollection > patTriggerObjectsStandAloneUnpacked( new TriggerObjectStandAloneCollection );

  for ( size_t iTrigObj = 0; iTrigObj < patTriggerObjectsStandAlone->size(); ++iTrigObj ) {
    TriggerObjectStandAlone patTriggerObjectStandAloneUnpacked( patTriggerObjectsStandAlone->at( iTrigObj ) );
    const edm::TriggerNames & names = iEvent.triggerNames( *triggerResults );
    patTriggerObjectStandAloneUnpacked.unpackPathNames( names );
    patTriggerObjectsStandAloneUnpacked->push_back( patTriggerObjectStandAloneUnpacked );
  }

  iEvent.put( patTriggerObjectsStandAloneUnpacked );
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PATTriggerObjectStandAloneUnpacker );
