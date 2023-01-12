// -*- C++ -*-
//
// Package:    WhatsItAnalyzer
// Class:      WhatsItAnalyzer
//
/**\class WhatsItAnalyzer WhatsItAnalyzer.cc test/WhatsItAnalyzer/src/WhatsItAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 19:13:25 EDT 2005
//
//

// system include files
#include <memory>
#include <iostream>
#include <vector>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "WhatsIt.h"
#include "GadgetRcd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"

//
// class decleration
//

namespace edmtest {

  class WhatsItAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit WhatsItAnalyzer(const edm::ParameterSet&);
    ~WhatsItAnalyzer() override;

    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    void getAndTest(edm::EventSetup const&,
                    edm::ESGetToken<WhatsIt, GadgetRcd> token,
                    int expectedValue,
                    const char* label);

    // ----------member data ---------------------------
    std::vector<int> expectedValues_;
    std::vector<std::pair<edm::ESGetToken<WhatsIt, GadgetRcd>, const char*>> tokenAndLabel_;
    unsigned int index_;
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
  WhatsItAnalyzer::WhatsItAnalyzer(const edm::ParameterSet& iConfig)
      : expectedValues_(iConfig.getUntrackedParameter<std::vector<int>>("expectedValues", std::vector<int>())),
        tokenAndLabel_(5),
        index_(0) {
    //now do what ever initialization is needed
    int i = 0;
    for (auto l : std::vector<const char*>({"", "A", "B", "C", "D"})) {
      tokenAndLabel_[i].first = esConsumes(edm::ESInputTag("", l));
      tokenAndLabel_[i++].second = l;
    }
  }

  WhatsItAnalyzer::~WhatsItAnalyzer() {
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
  }

  //
  // member functions
  //

  // ------------ method called to produce the data  ------------
  void WhatsItAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
    if (index_ < expectedValues_.size()) {
      int expectedValue = expectedValues_.at(index_);
      for (auto const& tl : tokenAndLabel_) {
        getAndTest(iSetup, tl.first, expectedValue, tl.second);
      }
      ++index_;
    }
  }

  void WhatsItAnalyzer::getAndTest(const edm::EventSetup& iSetup,
                                   edm::ESGetToken<WhatsIt, GadgetRcd> token,
                                   int expectedValue,
                                   const char* label) {
    auto const& v = iSetup.getData(token);
    if (expectedValue != v.a) {
      throw cms::Exception("TestFail") << label << ": expected value " << expectedValue << " but got " << v.a;
    }
  }
}  // namespace edmtest
using namespace edmtest;
//define this as a plug-in
DEFINE_FWK_MODULE(WhatsItAnalyzer);
