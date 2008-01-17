// -*- C++ -*-
//
// Package:    WFilter
// Class:      WFilter
// 
/**\class WFilter WFilter.cc HLTriggerOffline/Egamma/src/WFilter.cc

 Description: Applies genertor-level eta cuts to include electrons in Ecal fiducial volume

 Implementation:
     
*/
//
// Original Author:  Joshua Berger
//         Created:  Fri Jul 27 00:21:27 CEST 2007
// $Id: WFilter.cc,v 1.1 2007/09/14 19:05:50 jberger Exp $
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

class WFilter : public edm::EDFilter {
   public:
      explicit WFilter(const edm::ParameterSet&);
      ~WFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      bool accept;
      int nE_min_;
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
WFilter::WFilter(const edm::ParameterSet& iConfig)
{
  nE_min_ = iConfig.getParameter<int> ("nE_min");

   //now do what ever initialization is needed
   accept = false;
}


WFilter::~WFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
WFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   accept = false;

   Handle<HepMCProduct> mcProd;
   iEvent.getByLabel("source", mcProd);
   const HepMC::GenEvent * mcEvent = mcProd->GetEvent();

   int nE = 0;
   for(HepMC::GenEvent::particle_const_iterator mcpart = mcEvent->particles_begin(); mcpart != mcEvent->particles_end(); ++ mcpart ) {
     int id = (*mcpart)->pdg_id();
     if (abs(id) == 11) {
       for (HepMC::GenVertex::particles_in_const_iterator parent = (*mcpart)->production_vertex()->particles_in_const_begin(); parent != (*mcpart)->production_vertex()->particles_in_const_end(); ++ parent) {
	 std::cout<<"Parent ID: "<<(*parent)->pdg_id()<<std::endl;
	 if (abs((*parent)->pdg_id()) == 24) {
	   std::cout<<"Particle ID: "<<id<<std::endl;
	   nE++;
	 }
       }
     }
   }

   if (nE >= nE_min_) accept = true;

   return accept;
}

// ------------ method called once each job just before starting event loop  ------------
void 
WFilter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
WFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(WFilter);
