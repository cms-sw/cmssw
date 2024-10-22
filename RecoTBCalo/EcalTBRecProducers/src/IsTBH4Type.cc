//
// Original Author:  Pietro Govoni
//         Created:  Thu Aug 10 16:21:22 CEST 2006
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTBCalo/EcalTBRecProducers/interface/IsTBH4Type.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
IsTBH4Type::IsTBH4Type(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  eventHeaderCollection_ = iConfig.getParameter<std::string>("eventHeaderCollection");
  eventHeaderProducer_ = iConfig.getParameter<std::string>("eventHeaderProducer");
  typeToFlag_ = iConfig.getParameter<std::string>("typeToFlag");
  notFound_ = iConfig.getUntrackedParameter<bool>("ifHeaderNotFound", false);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool IsTBH4Type::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  Handle<EcalTBEventHeader> pEventHeader;
  const EcalTBEventHeader* evtHeader = nullptr;
  iEvent.getByLabel(eventHeaderProducer_, pEventHeader);
  if (!pEventHeader.isValid()) {
    edm::LogError("IsTBH4Type") << "Event Header collection not found";
  } else {
    evtHeader = pEventHeader.product();  // get a ptr to the product
  }

  if (!evtHeader)
    return notFound_;
  //   std::cout << "PIETRO " << evtHeader->eventType () << std::endl ;
  //   std::cout << "PIETRO " << (evtHeader->eventType () != typeToFlag_) << std::endl ;
  if (evtHeader->eventType() != typeToFlag_)
    return false;

  return true;
}
