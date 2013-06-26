// -*- C++ -*-
//
// Package:    PATHeavyIonProducer
// Class:      PATHeavyIonProducer
// 
/**\class PATHeavyIonProducer PATHeavyIonProducer.cc yetkin/PATHeavyIonProducer/src/PATHeavyIonProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Thu Aug 13 08:39:51 EDT 2009
// $Id: PATHeavyIonProducer.cc,v 1.4 2013/02/27 23:26:56 wmtan Exp $
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

#include "DataFormats/HeavyIonEvent/interface/HeavyIon.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "HepMC/HeavyIon.h"

using namespace std;

//
// class decleration
//

class PATHeavyIonProducer : public edm::EDProducer {
   public:
      explicit PATHeavyIonProducer(const edm::ParameterSet&);
      ~PATHeavyIonProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
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
PATHeavyIonProducer::PATHeavyIonProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   produces<pat::HeavyIon>();

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
  
}


PATHeavyIonProducer::~PATHeavyIonProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PATHeavyIonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

}

// ------------ method called once each job just before starting event loop  ------------
void 
PATHeavyIonProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PATHeavyIonProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(PATHeavyIonProducer);
