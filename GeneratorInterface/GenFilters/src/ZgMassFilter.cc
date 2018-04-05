#include "GeneratorInterface/GenFilters/interface/ZgMassFilter.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>
#include "TLorentzVector.h"

using namespace edm;
using namespace std;

ZgMassFilter::ZgMassFilter(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter("moduleLabel",edm::InputTag("generator","unsmeared")))),
minDileptonMass(iConfig.getUntrackedParameter("MinDileptonMass", 0.)),
minZgMass(iConfig.getUntrackedParameter("MinZgMass", 0.))
{
}

ZgMassFilter::~ZgMassFilter()
{
}

// ------------ method called to skim the data  ------------
bool ZgMassFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   bool accepted = false;
   Handle<HepMCProduct> evt;
   iEvent.getByToken(token_, evt);
   const HepMC::GenEvent * myGenEvent = evt->GetEvent();
     
   vector<TLorentzVector> Lepton; Lepton.clear();
   vector<TLorentzVector> Photon; Photon.clear();   

   for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	 p != myGenEvent->particles_end(); ++p ) {
     if ((*p)->status() == 1 && (abs((*p)->pdg_id()) == 11 || abs((*p)->pdg_id()) == 13 || abs((*p)->pdg_id()) == 15)) { 
       TLorentzVector LeptP((*p)->momentum().px(), (*p)->momentum().py(), (*p)->momentum().pz(), (*p)->momentum().e()); 
       Lepton.push_back(LeptP);
     }       
     if ( abs((*p)->pdg_id()) == 22 && (*p)->status() == 1) {
       TLorentzVector PhotP((*p)->momentum().px(), (*p)->momentum().py(), (*p)->momentum().pz(), (*p)->momentum().e()); 
       Photon.push_back(PhotP);
     }
   }
 
   if (Lepton.size() > 1 && (Lepton[0]+Lepton[1]).M() > minDileptonMass ) {
     if ((Lepton[0]+Lepton[1]+Photon[0]).M() > minZgMass) {
       accepted = true; 
     }
   }

   return accepted;    
}
