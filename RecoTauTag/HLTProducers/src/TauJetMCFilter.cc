#include "RecoTauTag/HLTProducers/interface/TauJetMCFilter.h"
#include "Math/GenVector/VectorUtil.h"

//
// class decleration
//
using namespace edm;
using namespace HepMC;
using namespace std;

TauJetMCFilter::TauJetMCFilter(const edm::ParameterSet& iConfig)
{
  genParticles = iConfig.getParameter<InputTag>("GenParticles");
  mEtaMax = iConfig.getParameter<double>("EtaTauMax");
  mEtaMin = iConfig.getParameter<double>("EtaTauMin");
  mEtTau = iConfig.getParameter<double>("EtTau");
  
}

TauJetMCFilter::~TauJetMCFilter(){ }

HepMC::GenParticle * TauJetMCFilter::findParticle(const GenPartVect genPartVect,const int requested_id)
{
  for (GenPartVectIt p = genPartVect.begin(); p != genPartVect.end(); p++)
    {
      if (requested_id == (*p)->pdg_id()) return *p;
    }
  return 0;
}

bool TauJetMCFilter::filter(edm::Event& iEvent, const edm::EventSetup& iES)
{

  edm::Handle<HepMCProduct> evt;
  iEvent.getByLabel(genParticles, evt);

  HepMC::GenEvent * generated_event = new HepMC::GenEvent(*(evt->GetEvent()));

  bool event_passed = true;
  HepMC::GenEvent::particle_iterator p;
  for (p = generated_event->particles_begin(); p != generated_event->particles_end(); p++) {
    if(abs((*p)->pdg_id()) == 15) { //Searches for Taus

      HepLorentzVector partMom = ( *p )->momentum() ;
      if(partMom.et() < mEtTau) event_passed = false;
      
      vector< GenParticle * > child = (*p)->listChildren();
      for (GenPartVectIt z = child.begin(); z != child.end(); z++)
	{
	  
	  if(abs((*z)->pdg_id()) == 11 || abs((*z)->pdg_id()) == 13) //See if there are muons or electrons among the children
	    event_passed = false;
	}
    }
    
  }

  delete generated_event; 
  return event_passed;
  
  
}
