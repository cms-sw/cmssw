#include <iostream>
#include "DataFormats/TestObjects/interface/OtherThing.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"


namespace edmtest {
  
  class OtherThingRefComparer : public edm::stream::EDAnalyzer<> {
  public:
    
    explicit OtherThingRefComparer(edm::ParameterSet const& pset);
    
    virtual void analyze(edm::Event const& e, edm::EventSetup const& c) override;
    
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  private:
    edm::EDGetTokenT<OtherThingCollection> token1_;
    edm::EDGetTokenT<OtherThingCollection> token2_;
  };
  
  OtherThingRefComparer::OtherThingRefComparer(edm::ParameterSet const& pset) :
  token1_(consumes<OtherThingCollection>(pset.getUntrackedParameter<edm::InputTag>("first"))),
  token2_(consumes<OtherThingCollection>(pset.getUntrackedParameter<edm::InputTag>("second")))
 {
  }

  void OtherThingRefComparer::analyze(edm::Event const& e, edm::EventSetup const&) {
    edm::Handle<OtherThingCollection> handle1_;
    e.getByToken(token1_,handle1_);
    edm::Handle<OtherThingCollection> handle2_;
    e.getByToken(token2_,handle2_);
    
    assert(handle1_->size() == handle2_->size());

    {
      auto iter2 = handle2_->begin();
      for(auto const& o1: *handle1_) {
        if(o1.ref != iter2->ref) {
          throw cms::Exception("RefCompareFailure")<<"edm::Refs are not equal"<< o1.ref.id()<<" "<<iter2->ref.id();
        }
        ++iter2;
      }
    }

    {
      auto iter2 = handle2_->begin();
      for(auto const& o1: *handle1_) {
        if(o1.ptr != iter2->ptr) {
          throw cms::Exception("RefCompareFailure")<<"edm::Ptrs are not equal"<<o1.ptr.id()<<" "<<iter2->ptr.id();
        }
        ++iter2;
      }
    }

    
  }

  void OtherThingRefComparer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<edm::InputTag>("first",edm::InputTag("OtherThing","testUserTag"))->setComment("Where to get the first OtherThingCollection");
    desc.addUntracked<edm::InputTag>("second",edm::InputTag("OtherThing","testUserTag"))->setComment("Where to get the second OtherThingCollection");
    descriptions.add("otherThingRefComparer", desc);
  }

}
using edmtest::OtherThingRefComparer;
DEFINE_FWK_MODULE(OtherThingRefComparer);
