// -*- C++ -*-
//
// Package:    GeneratorInterface/LHEInterface
// Class:      LHESlimmedWeightsProducer
// 
/**\class LHESlimmedWeightsProducer LHESlimmedWeightsProducer.cc GeneratorInterface/LHESlimmedWeightsProducer/plugins/LHESlimmedWeightsProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Pietro Govoni
//         Created:  Mon, 19 Oct 2015 13:55:04 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


//---- to get weights
// #include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
// #include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
// #include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"


//---- for LHE information
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProductLite.h"
// #include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
// #include "SimDataFormats/GeneratorProducts/interface/LHEXMLStringProduct.h"


using namespace edm ;
using namespace std ;

//
// class declaration
//

class LHESlimmedWeightsProducer : public edm::EDProducer {
   public:
      explicit LHESlimmedWeightsProducer(const edm::ParameterSet&) ;
      ~LHESlimmedWeightsProducer() ;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) ;

   private:
      virtual void beginJob() override ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override ;
      virtual void endJob() override ;
      
      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override ;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override ;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override ;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override ;

      // ----------member data ---------------------------
} ;

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
LHESlimmedWeightsProducer::LHESlimmedWeightsProducer(const edm::ParameterSet& iConfig)
{
   produces<LHEEventProductLite>("");
}


LHESlimmedWeightsProducer::~LHESlimmedWeightsProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
LHESlimmedWeightsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<LHEEventProduct> LHEEventProductHandle ;
  iEvent.getByLabel("externalLHEProducer", LHEEventProductHandle) ;

  std::auto_ptr<LHEEventProductLite> result (new LHEEventProductLite (LHEEventProductHandle.product ())) ;
        
  result->setNpLO (LHEEventProductHandle.product ()->npLO ()) ;
  result->setNpNLO (LHEEventProductHandle.product ()->npNLO ()) ;
 
  iEvent.put (result) ;
  return ;
}

// ------------ method called once each job just before starting event loop  ------------
void 
LHESlimmedWeightsProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
LHESlimmedWeightsProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
LHESlimmedWeightsProducer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
LHESlimmedWeightsProducer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
LHESlimmedWeightsProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
LHESlimmedWeightsProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
LHESlimmedWeightsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc ;
  desc.setUnknown() ;
  descriptions.addDefault(desc) ;
}

//define this as a plug-in
DEFINE_FWK_MODULE(LHESlimmedWeightsProducer) ;
