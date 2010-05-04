// -*- C++ -*-
//
// Package:    GenHIEventProducer
// Class:      GenHIEventProducer
// 
/**\class GenHIEventProducer GenHIEventProducer.cc yetkin/GenHIEventProducer/src/GenHIEventProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Thu Aug 13 08:39:51 EDT 2009
// $Id: GenHIEventProducer.cc,v 1.2 2010/02/20 21:00:22 wmtan Exp $
//
//


// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"

#include "HepMC/HeavyIon.h"

using namespace std;

//
// class decleration
//

class GenHIEventProducer : public edm::EDProducer {
   public:
      explicit GenHIEventProducer(const edm::ParameterSet&);
      ~GenHIEventProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

   bool doMC_;
   bool doReco_;
   std::vector<std::string> hepmcSrc_;
   edm::InputTag centSrc_;
   edm::InputTag evtPlaneSrc_;

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
GenHIEventProducer::GenHIEventProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   //   produces<pat::HeavyIon>();
   produces<edm::GenHIEvent>();

   //now do what ever other initialization is needed
   doReco_ = iConfig.getParameter<bool>("doReco");
   if(doReco_){
      centSrc_ = iConfig.getParameter<edm::InputTag>("centrality");
      evtPlaneSrc_ = iConfig.getParameter<edm::InputTag>("evtPlane");
   }

   doMC_ = iConfig.getParameter<bool>("doMC");
   if(doMC_){
      hepmcSrc_ = iConfig.getParameter<std::vector<std::string> >("generators");
   }
  
   doReco_ = false;
   doMC_ = true;
}


GenHIEventProducer::~GenHIEventProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GenHIEventProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   double b = -1;
   int npart = -1;
   int ncoll = 0;
   int nhard = 0;
   double phi = 0;

      for(size_t ihep = 0; ihep < hepmcSrc_.size(); ++ihep){
	 Handle<edm::HepMCProduct> hepmc;
	 iEvent.getByLabel(hepmcSrc_[ihep],hepmc);
	 const HepMC::HeavyIon* hi = hepmc->GetEvent()->heavy_ion();
	 if(hi){
	    ncoll = ncoll + hi->Ncoll();
	    nhard = nhard + hi->Ncoll_hard();
	    int np = hi->Npart_proj() + hi->Npart_targ();
	    if(np >= 0){
	       npart = np;
	       b = hi->impact_parameter();
	       phi = hi->event_plane_angle();
	    }
	 }
      }
      std::auto_ptr<edm::GenHIEvent> pGenHI(new edm::GenHIEvent(b,
								npart,
								ncoll,
								nhard,
								phi));



      
      iEvent.put(pGenHI);

}

// ------------ method called once each job just before starting event loop  ------------
void 
GenHIEventProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
GenHIEventProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(GenHIEventProducer);
