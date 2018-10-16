// -*- C++ -*-
//
// Package:    WhatsItESProducer
// Class:      WhatsItESProducer
//
/**\class WhatsItESProducer WhatsItESProducer.h test/WhatsItESProducer/interface/WhatsItESProducer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 14:33:04 EDT 2005
//
//


// system include files
#include <memory>
#include <optional>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Integration/test/WhatsIt.h"
#include "FWCore/Integration/test/Doodad.h"
#include "FWCore/Integration/test/GadgetRcd.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

//
// class decleration
//
namespace edmtest {

class WhatsItESProducer : public edm::ESProducer {
   public:
      WhatsItESProducer(edm::ParameterSet const& pset);
      ~WhatsItESProducer();

      typedef std::unique_ptr<WhatsIt> ReturnType;
      typedef std::unique_ptr<const WhatsIt> ReturnTypeA;
      typedef std::shared_ptr<WhatsIt> ReturnTypeB;
      typedef std::shared_ptr<const WhatsIt> ReturnTypeC;
      typedef std::optional<WhatsIt> ReturnTypeD;

      ReturnType produce(const GadgetRcd &);
      ReturnTypeA produceA(const GadgetRcd &);
      ReturnTypeB produceB(const GadgetRcd &);
      ReturnTypeC produceC(const GadgetRcd &);
      ReturnTypeD produceD(const GadgetRcd &);

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      // ----------member data ---------------------------
      edm::ESGetTokenT<edmtest::Doodad> token_;
      edm::ESGetTokenT<edmtest::Doodad> tokenA_;
      edm::ESGetTokenT<edmtest::Doodad> tokenB_;
      edm::ESGetTokenT<edmtest::Doodad> tokenC_;
      edm::ESGetTokenT<edmtest::Doodad> tokenD_;
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
WhatsItESProducer::WhatsItESProducer(edm::ParameterSet const& pset)
{
  if (pset.getUntrackedParameter<bool>("test", true)) {
     throw edm::Exception(edm::errors::Configuration, "Something is wrong with ESProducer validation\n")
       << "Or the test configuration parameter was set true (it should never be true unless you want this exception)\n";
  }

  //the following line is needed to tell the framework what
  // data is being produced
  auto collector = setWhatProduced(this);
  auto collectorA = setWhatProduced(this, &WhatsItESProducer::produceA, edm::es::Label("A"));
  auto collectorB = setWhatProduced(this, &WhatsItESProducer::produceB, edm::es::Label("B"));
  auto collectorC = setWhatProduced(this, &WhatsItESProducer::produceC, edm::es::Label("C"));
  auto collectorD = setWhatProduced(this, &WhatsItESProducer::produceD, edm::es::Label("D"));

  //now do what ever other initialization is needed
  auto const data_label = pset.exists("doodadLabel") ? pset.getParameter<std::string>("doodadLabel"): std::string{};
  token_ = collector.consumes<edmtest::Doodad>(edm::ESInputTag{"", data_label});
  tokenA_ = collectorA.consumes<edmtest::Doodad>(edm::ESInputTag{"", data_label});
  tokenB_ = collectorB.consumes<edmtest::Doodad>(edm::ESInputTag{"", data_label});
  tokenC_ = collectorC.consumes<edmtest::Doodad>(edm::ESInputTag{"", data_label});
  tokenD_ = collectorD.consumes<edmtest::Doodad>(edm::ESInputTag{"", data_label});
}

WhatsItESProducer::~WhatsItESProducer()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
WhatsItESProducer::ReturnType
WhatsItESProducer::produce(const GadgetRcd& iRecord)
{
   edm::ESHandle<Doodad> doodad;
   iRecord.get(token_, doodad);
   auto pWhatsIt = std::make_unique<WhatsIt>() ;
   pWhatsIt->a = doodad->a;
   return pWhatsIt ;
}

WhatsItESProducer::ReturnTypeA
WhatsItESProducer::produceA(const GadgetRcd& iRecord)
{
   edm::ESHandle<Doodad> doodad;
   iRecord.get(tokenA_, doodad);
   auto pWhatsIt = std::make_unique<WhatsIt>() ;
   pWhatsIt->a = doodad->a;
   return pWhatsIt ;
}

WhatsItESProducer::ReturnTypeB
WhatsItESProducer::produceB(const GadgetRcd& iRecord)
{
   edm::ESHandle<Doodad> doodad;
   iRecord.get(tokenB_ ,doodad);
   auto pWhatsIt = std::make_shared<WhatsIt>() ;
   pWhatsIt->a = doodad->a;
   return pWhatsIt ;
}

WhatsItESProducer::ReturnTypeC
WhatsItESProducer::produceC(const GadgetRcd& iRecord)
{
   edm::ESHandle<Doodad> doodad;
   iRecord.get(tokenC_, doodad);
   auto pWhatsIt = std::make_shared<WhatsIt>() ;
   pWhatsIt->a = doodad->a;
   return pWhatsIt ;
}

WhatsItESProducer::ReturnTypeD
WhatsItESProducer::produceD(const GadgetRcd& iRecord)
{
   edm::ESHandle<Doodad> doodad;
   iRecord.get(tokenD_, doodad);
   auto pWhatsIt = std::make_optional<WhatsIt>() ;
   pWhatsIt->a = doodad->a;
   return pWhatsIt ;
}

void
WhatsItESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addOptional<std::string>("doodadLabel");
  desc.addUntracked<bool>("test", false)->
    setComment("This parameter exists only to test the parameter set validation for ESSources");
  descriptions.add("WhatsItESProducer", desc);
}
}
using namespace edmtest;
//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(WhatsItESProducer);
