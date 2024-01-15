// -*- C++ -*-
//
// Package:    HLTrigger/TestBXVectorRefProducer
// Class:      TestBXVectorRefProducer
//
/**\class TestBXVectorRefProducer TestBXVectorRefProducer.cc HLTrigger/TestBXVectorRefProducer/plugins/TestBXVectorRefProducer.cc

 Description: Simple testing producer to test storing of Ref<BXVector>  (example of <l1t::JetRef>) in the Event.

 Implementation:
     Pick up the BXVector<l1t::Jet> from the event and try to store the Refs back into the Event.
*/
//
// Original Author:  Vladimir Rekovic
//         Created:  Fri, 12 Feb 2016 09:56:04 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

//#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/Jet.h"

//
// class declaration
//

class TestBXVectorRefProducer : public edm::stream::EDProducer<> {
public:
  explicit TestBXVectorRefProducer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  bool doRefs_;
  edm::InputTag src_;
  edm::EDGetTokenT<l1t::JetBxCollection> token_;
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
TestBXVectorRefProducer::TestBXVectorRefProducer(const edm::ParameterSet& iConfig) {
  //now do what ever other initialization is needed
  src_ = iConfig.getParameter<edm::InputTag>("src");
  doRefs_ = iConfig.getParameter<bool>("doRefs");
  token_ = consumes<l1t::JetBxCollection>(src_);

  //register your products
  produces<vector<int>>("jetPt").setBranchAlias("jetPt");

  if (doRefs_) {
    produces<l1t::JetRefVector>("l1tJetRef").setBranchAlias("l1tJetRef");
  }
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void TestBXVectorRefProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  unique_ptr<vector<int>> jetMom(new vector<int>);
  unique_ptr<l1t::JetRefVector> jetRef(new l1t::JetRefVector);

  // retrieve the tracks
  Handle<l1t::JetBxCollection> jets;
  iEvent.getByToken(token_, jets);
  if (!jets.isValid())
    return;

  const int size = jets->size();
  jetMom->reserve(size);

  l1t::JetBxCollection::const_iterator iter;

  for (iter = jets->begin(0); iter != jets->end(0); ++iter) {
    jetMom->push_back(iter->pt());

    l1t::JetRef myref(jets, jets->key(iter));
    jetRef->push_back(myref);

  }  // end for

  iEvent.put(std::move(jetMom), "jetPt");

  if (doRefs_)
    iEvent.put(std::move(jetRef), "l1tJetRef");

  return;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TestBXVectorRefProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

// declare this class as a framework plugin
DEFINE_FWK_MODULE(TestBXVectorRefProducer);
