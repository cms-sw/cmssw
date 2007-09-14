// -*- C++ -*-
//
// Package:    MCEtaFilter
// Class:      MCEtaFilter
// 
/**\class MCEtaFilter MCEtaFilter.cc HLTriggerOffline/Egamma/src/MCEtaFilter.cc

 Description: Applies genertor-level eta cuts to include electrons in Ecal fiducial volume

 Implementation:
     
*/
//
// Original Author:  Joshua Berger
//         Created:  Fri Jul 27 00:21:27 CEST 2007
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
//
// class declaration
//

class MCEtaFilter : public edm::EDFilter {
   public:
      explicit MCEtaFilter(const edm::ParameterSet&);
      ~MCEtaFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
     edm::InputTag candTag_;
     int ncandcut_;
     int pathId_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MCEtaFilter::MCEtaFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  ncandcut_ = iConfig.getParameter<int> ("ncandcut");
  pathId_ = iConfig.getParameter<int> ("id");
}


MCEtaFilter::~MCEtaFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
MCEtaFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<HepMCProduct> mcProd;
   iEvent.getByLabel(candTag_, mcProd);
   const HepMC::GenEvent * mcEvent = mcProd->GetEvent();

   int candidates = 0;
   bool accept = false;
   int electronN = 0;
   for(HepMC::GenEvent::particle_const_iterator mcpart = mcEvent->particles_begin(); mcpart != mcEvent->particles_end(); ++ mcpart ) {
     double eta = (*mcpart)->momentum().eta();
     int id = (*mcpart)->pdg_id();
     int status = (*mcpart)->status();
     electronN++;
     if ((pathId_ == -1 || abs(id) == pathId_) && status == 3) {
       if (fabs(eta) < 2.5 && (fabs(eta) < 1.4442 || fabs(eta) > 1.566)) {
         candidates++;
       }
     }
   }

   if (candidates >= ncandcut_) accept = true;

   return accept;
}

// ------------ method called once each job just before starting event loop  ------------
void 
MCEtaFilter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MCEtaFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MCEtaFilter);
