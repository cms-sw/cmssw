//
// Select the partons status 2 and 3 for MC Jet Flavour 
// Author: Attilio
// Date: 10.10.2007
//
    
//=======================================================================

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"  
//#include "DataFormats/Candidate/interface/CandidateFwd.h"
//#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
      
#include <memory>
#include <string>
#include <iostream>
#include <vector>
         
using namespace std;
using namespace reco;
using namespace edm;

class PartonSelector : public edm::EDProducer 
{
  public:
    PartonSelector( const edm::ParameterSet & );
    ~PartonSelector();

  private:

    virtual void produce(edm::Event&, const edm::EventSetup& ); 
    bool withLeptons;
};
//=========================================================================

PartonSelector::PartonSelector( const edm::ParameterSet& iConfig )
{ 
    produces<reco::GenParticleRefVector>();
    withLeptons         = iConfig.getParameter<bool>("withLeptons");   
}

//=========================================================================

PartonSelector::~PartonSelector()
{
}

// ------------ method called to produce the data  ------------

void PartonSelector::produce( Event& iEvent, const EventSetup& iEs )
{
  
  //edm::Handle <reco::CandidateView> particles;
  edm::Handle <reco::GenParticleCollection> particles;
  iEvent.getByLabel ("genParticles", particles );  
  edm::LogVerbatim("PartonSelector") << "=== GenParticle size:" << particles->size();
  int nPart=0; 

  auto_ptr<GenParticleRefVector> thePartons ( new GenParticleRefVector);  
  
  for (size_t m = 0; m < particles->size(); m++) {
    
    // Don't take into account first 6 particles in generator list
    if (m<6) continue;    

    const GenParticle & aParticle = (*particles)[ m ];
    bool isAParton = false;
    bool isALepton = false;
    int flavour = abs(aParticle.pdgId());
    if(flavour == 1 ||
       flavour == 2 ||
       flavour == 3 ||
       flavour == 4 ||
       flavour == 5 ||  
       flavour == 21 ) isAParton = true;
    if(flavour == 11 ||
       flavour == 12 ||
       flavour == 13 ||
       flavour == 14 ||
       flavour == 15 ||
       flavour == 16 ) isALepton = true;

    //Add Partons status 3
    if( aParticle.status() == 3 && isAParton ) {
      thePartons->push_back( GenParticleRef( particles, m ) );
      nPart++;
    }

    //Add Partons stauts 2
    if(   aParticle.numberOfDaughters() > 0  && 
        ( aParticle.daughter(0)->pdgId() == 91 || aParticle.daughter(0)->pdgId() == 92 ) ) {
      thePartons->push_back( GenParticleRef( particles, m ) );
      nPart++;
    }

    //Add Leptons
    // Here you have to decide what to do with taus....
    // Now all leptons, including e and mu from leptonic tau decays, are added
    if( withLeptons && aParticle.status() == 3 && isALepton ) {
      thePartons->push_back( GenParticleRef( particles, m ) );
      nPart++;
    }
  }
  
  edm::LogVerbatim("PartonSelector") << "=== GenParticle selected:" << nPart;  
  iEvent.put( thePartons );

}

DEFINE_FWK_MODULE(PartonSelector);

