#include "RecoTauTag/HLTProducers/interface/TauJetMCFilter.h"

#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "Math/GenVector/VectorUtil.h"

//
// class decleration
//
using namespace edm;
using namespace HepMC;
using namespace std;
 using namespace l1extra;

TauJetMCFilter::TauJetMCFilter(const edm::ParameterSet& iConfig)
{
  genParticles = iConfig.getParameter<InputTag>("GenParticles");
  mEtaMax = iConfig.getParameter<double>("EtaTauMax");
  mEtaMin = iConfig.getParameter<double>("EtaTauMin");
  mEtTau = iConfig.getParameter<double>("EtTau");
  tauParticles = iConfig.getParameter<InputTag>("TauParticles");

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
  //
  Handle< L1JetParticleCollection > tauColl ; 
  iEvent.getByLabel( tauParticles, tauColl );
  const L1JetParticleCollection & myL1Tau  = *(tauColl.product());
  //
  
  edm::Handle<HepMCProduct> evt;
  iEvent.getByLabel(genParticles, evt);

  HepMC::GenEvent * generated_event = new HepMC::GenEvent(*(evt->GetEvent()));

  int nTauMatched = 0;
  bool event_passed = true;
  HepMC::GenEvent::particle_iterator p;
  for (p = generated_event->particles_begin(); p != generated_event->particles_end(); p++) {
    if(abs((*p)->pdg_id()) == 15) { //Searches for Taus

      HepLorentzVector partMom = ( *p )->momentum() ;
      if(partMom.et() < mEtTau) event_passed = false;
      if(fabs(partMom.eta()) > mEtaMax) event_passed = false;

      vector< GenParticle * > child = (*p)->listChildren();
      for (GenPartVectIt z = child.begin(); z != child.end(); z++)
	{
	  if(abs((*z)->pdg_id()) == 11 || abs((*z)->pdg_id()) == 13) //See if there are muons or electrons among the children
	    event_passed = false;
	}

      
      /*
      if(event_passed)
	{
	  L1JetParticleCollection::const_iterator l1Tau = myL1Tau.begin();
	  for(;l1Tau != myL1Tau.end();l1Tau++)
	    {
	      math::XYZVector candDir((*p)->momentum().x(),(*p)->momentum().y(),(*p)->momentum().z());
	      double deltaR = ROOT::Math::VectorUtil::DeltaR((*l1Tau).momentum(), candDir);
	      if(deltaR < 0.15) nTauMatched++;


	    }

	}
      */
    }
    
  }
  if(event_passed) cout <<"# Matched Taus "<<nTauMatched<<endl;
  delete generated_event; 
  return event_passed;
  
  
}
