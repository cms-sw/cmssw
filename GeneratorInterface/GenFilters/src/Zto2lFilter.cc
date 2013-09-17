// -*- C++ -*-
//
// Package:    Zto2lFilter
// Class:      Zto2lFilter
// 
/**\class Zto2lFilter Zto2lFilter.cc Zbb/Zto2lFilter/src/Zto2lFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Aruna Nayak
//         Created:  Thu Aug 23 11:37:45 CEST 2007
//
//


// system include files
#include <memory>
#include "GeneratorInterface/GenFilters/interface/Zto2lFilter.h"

#include <vector>
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "TLorentzVector.h"

//
// constants, enums and typedefs
//

using namespace std; 
using namespace edm;
//
// static data member definitions
//

//
// constructors and destructor
//
Zto2lFilter::Zto2lFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  fLabel_ = iConfig.getUntrackedParameter("moduleLabel",std::string("generator"));
  maxEtaLepton_ = iConfig.getUntrackedParameter<double>("MaxEtaLepton");
  minInvariantMass_ = iConfig.getUntrackedParameter<double>("MindiLeptonInvariantMass");
  
}


Zto2lFilter::~Zto2lFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
Zto2lFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   bool accept = false;

   Handle<HepMCProduct> EvtHandle ;
   iEvent.getByLabel( fLabel_, EvtHandle ) ;
   const HepMC::GenEvent* evt = EvtHandle->GetEvent();
   
   vector<TLorentzVector> Lepton; Lepton.clear();
   for (HepMC::GenEvent::particle_const_iterator p = evt->particles_begin();
	p != evt->particles_end(); ++p) {
     if((*p)->status()==3){
       if ( abs((*p)->pdg_id()) == 11 || abs((*p)->pdg_id()) == 13 || abs((*p)->pdg_id()) == 15  ){
	 if(fabs((*p)->momentum().eta()) < maxEtaLepton_){
	   TLorentzVector LeptP((*p)->momentum().px(), (*p)->momentum().py(), (*p)->momentum().pz(), (*p)->momentum().e()); 
	   Lepton.push_back(LeptP);
	 }
       }
     }
   }
   if(Lepton.size() == 2){
     if((Lepton[0]+Lepton[1]).M() > minInvariantMass_ )accept = true;
   }
   //delete evt;
   return accept;
}

// ------------ method called once each job just before starting event loop  ------------

// ------------ method called once each job just after ending the event loop  ------------
void 
Zto2lFilter::endJob() {
}

//define this as a plug-in
//DEFINE_FWK_MODULE(Zto2lFilter);
