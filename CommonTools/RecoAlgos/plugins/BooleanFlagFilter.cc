// -*- C++ -*-
//
// Package:    CommonTools/RecoAlgos
// Class:      BooleanFlagFilter
//
/**\class BooleanFlagFilter BooleanFlagFilter.cc CommonTools/RecoAlgos/plugins/BooleanFlagFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Fri, 20 Mar 2015 08:05:20 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class BooleanFlagFilter : public edm::global::EDFilter<> {
public:
  explicit BooleanFlagFilter(const edm::ParameterSet&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<bool> inputToken_;
  bool reverse_;
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
BooleanFlagFilter::BooleanFlagFilter(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  inputToken_ = consumes<bool>(iConfig.getParameter<edm::InputTag>("inputLabel"));
  reverse_ = iConfig.getParameter<bool>("reverseDecision");
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool BooleanFlagFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  Handle<bool> pIn;
  iEvent.getByToken(inputToken_, pIn);
  if (!pIn.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound) << " could not find requested flag\n";
    return true;
  }

  bool result = *pIn;
  if (reverse_)
    result = !result;

  return result;
}

//define this as a plug-in
DEFINE_FWK_MODULE(BooleanFlagFilter);
