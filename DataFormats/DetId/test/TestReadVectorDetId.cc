// -*- C++ -*-
//
// Package:    DataFormats/DetId
// Class:      TestReadVectorDetId
//
/**\class edmtest::TestReadVectorDetId
  Description: Used as part of tests that ensure the std::vector<DetId>
  raw data format can be persistently written and in a subsequent process
  read. First, this is done using the current release version for writing
  and reading. In addition, the output file of the write process should
  be saved permanently each time the VectorDetId persistent data
  format changes. In unit tests, we read each of those saved files to verify
  that the current releases can read older versions of the data format.
*/
// Original Author:  W. David Dagenhart
//         Created:  25 September 2023

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <vector>

namespace edmtest {

  class TestReadVectorDetId : public edm::global::EDAnalyzer<> {
  public:
    TestReadVectorDetId(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    void throwWithMessage(const char*) const;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    // This expected value is meaningless other than we use it
    // to check that values read from persistent storage match the values
    // we know were written.
    unsigned int expectedTestValue_;

    edm::EDGetTokenT<std::vector<DetId>> collectionToken_;
  };

  TestReadVectorDetId::TestReadVectorDetId(edm::ParameterSet const& iPSet)
      : expectedTestValue_(iPSet.getParameter<unsigned int>("expectedTestValue")),
        collectionToken_(consumes(iPSet.getParameter<edm::InputTag>("collectionTag"))) {}

  void TestReadVectorDetId::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    auto const& vectorDetIds = iEvent.get(collectionToken_);

    unsigned int expectedNumberOfDetIds = (iEvent.id().event() - 1) % 10;
    unsigned int expectedDetId = expectedTestValue_ + iEvent.id().event();
    unsigned int numberOfDetIds = 0;
    for (const auto& detId : vectorDetIds) {
      ++numberOfDetIds;
      expectedDetId += iEvent.id().event();
      if (detId.rawId() != expectedDetId) {
        throwWithMessage("DetId in vector of DetIds does not have expected value");
      }
    }
    if (numberOfDetIds != expectedNumberOfDetIds) {
      throwWithMessage("Number of DetIds does not match expected value");
    }
  }

  void TestReadVectorDetId::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestReadVectorDetId::analyze, " << msg;
  }

  void TestReadVectorDetId::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<unsigned int>("expectedTestValue");
    desc.add<edm::InputTag>("collectionTag");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestReadVectorDetId;
DEFINE_FWK_MODULE(TestReadVectorDetId);
