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
// $Id: GenHIEventProducer.cc,v 1.1 2010/05/04 16:05:50 yilmaz Exp $
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
      virtual void produce(edm::Event&, const edm::EventSetup&);
   std::vector<std::string> hepmcSrc_;
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
   produces<edm::GenHIEvent>();
   hepmcSrc_ = iConfig.getParameter<std::vector<std::string> >("generators");
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

//define this as a plug-in
DEFINE_FWK_MODULE(GenHIEventProducer);
