//
// $Id: PATL1Producer.cc,v 1.1.2.3 2008/04/24 16:23:13 vadler Exp $
//


#include "PhysicsTools/PatAlgos/plugins/PATL1Producer.h"

#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"

#include "DataFormats/PatCandidates/interface/TriggerPrimitive.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


using namespace pat;
using namespace l1extra;
using namespace edm;
using namespace std;


PATL1Producer::PATL1Producer( const ParameterSet& iConfig ) :
  // initialize
  particleMaps_ (iConfig.getParameter<InputTag>( "particleMaps" ) ),
  triggerName_  (iConfig.getParameter<string>  ( "triggerName" ) ),
  objectType_   (iConfig.getParameter<string>  ( "objectType" ) )
{
  produces<TriggerPrimitiveCollection>();
}


PATL1Producer::~PATL1Producer()
{
}


void PATL1Producer::produce( Event& iEvent, const EventSetup& iSetup )
{
  auto_ptr<TriggerPrimitiveCollection> patL1Candidates( new TriggerPrimitiveCollection );
  Handle<L1ParticleMapCollection> particleMaps;
  iEvent.getByLabel( particleMaps_, particleMaps );
  const L1ParticleMap::L1TriggerType triggerType = L1ParticleMap::triggerType( triggerName_ );
  if ( triggerType == L1ParticleMap::kNumOfL1TriggerTypes ) {
    LogDebug( "wrongL1TriggerName" ) << "PATL1Producer: The L1 trigger name " << triggerName_ << " is not known in this event!";
  } else {
    const L1ParticleMap& particleMap = particleMaps->at( triggerType );
    if ( ! particleMap.triggerDecision() ) {
      LogDebug( "notacceptL1Trigger" ) << "PATL1Producer: The L1 trigger " << triggerName_ << " did not accept this event!";
    } else {
      if ( objectType_ == "em" ) { // isolated or non-isolated (for electrons and photons)
        const L1EmParticleVectorRef& triggeredObjects = particleMap.emParticles();
        // loop over L1 trigger objects and store L1 trigger candidates
        for ( unsigned int iTriggeredObjects = 0; iTriggeredObjects < triggeredObjects.size(); ++iTriggeredObjects ) {
          auto_ptr<TriggerPrimitive> ptr( new TriggerPrimitive( triggeredObjects[ iTriggeredObjects ]->p4(), triggerName_, objectType_ ) );
          patL1Candidates->push_back( ptr );        
        }
      } else if ( objectType_ == "muon" ) {
        const L1MuonParticleVectorRef& triggeredObjects = particleMap.muonParticles();
        // loop over L1 trigger objects and store L1 trigger candidates
        for ( unsigned int iTriggeredObjects = 0; iTriggeredObjects < triggeredObjects.size(); ++iTriggeredObjects ) {
          auto_ptr<TriggerPrimitive> ptr( new TriggerPrimitive( triggeredObjects[ iTriggeredObjects ]->p4(), triggerName_, objectType_ ) );
          patL1Candidates->push_back( ptr );        
        }
      } else if ( objectType_ == "jet" ) { // central or forward (for jets) or tau
        const L1JetParticleVectorRef& triggeredObjects = particleMap.jetParticles();
        // loop over L1 trigger objects and store L1 trigger candidates
        for ( unsigned int iTriggeredObjects = 0; iTriggeredObjects < triggeredObjects.size(); ++iTriggeredObjects ) {
          auto_ptr<TriggerPrimitive> ptr( new TriggerPrimitive( triggeredObjects[ iTriggeredObjects ]->p4(), triggerName_, objectType_ ) );
          patL1Candidates->push_back( ptr );        
        }
      } else if ( objectType_ == "met" ) {
        const L1EtMissParticleRefProd& triggeredObject = particleMap.etMissParticle();
        // store L1 trigger candidates
        auto_ptr<TriggerPrimitive> ptr( new TriggerPrimitive( triggeredObject->p4(), triggerName_, objectType_ ) );
        patL1Candidates->push_back( ptr );        
      } else { // wrong input to configurable
        LogDebug( "wrongL1Object" ) << "PATL1Producer: The L1 object type " << objectType_ << "does not exist!";
      }
    }
  }
  iEvent.put( patL1Candidates );
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATL1Producer);
