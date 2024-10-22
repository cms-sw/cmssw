// -*- C++ -*-
//
// Package:    DataFormats/FEDRawData
// Class:      TestReadFEDRawDataCollection
//
/**\class edmtest::TestReadFEDRawDataCollection
  Description: Used as part of tests that ensure the FEDRawDataCollection
  data format can be persistently written and in a subsequent process
  read. First, this is done using the current release version for writing
  and reading. In addition, the output file of the write process should
  be saved permanently each time the FEDRawDataCollection persistent data
  format changes. In unit tests, we read each of those saved files to verify
  that the current releases can read older versions of the data format.
*/
// Original Author:  W. David Dagenhart
//         Created:  1 May 2023

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
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

  class TestReadFEDRawDataCollection : public edm::global::EDAnalyzer<> {
  public:
    TestReadFEDRawDataCollection(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    void throwWithMessage(const char*) const;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    // Two FEDRawData elements should be enough to verify we can read
    // and write the whole collection. I arbitrarily chose elements
    // 0 and 3 of the Collection. Values are meaningless, we just
    // verify what we read matches what we wrote. For purposes of
    // this test that is enough.
    std::vector<unsigned int> expectedFEDData0_;
    std::vector<unsigned int> expectedFEDData3_;
    edm::EDGetTokenT<FEDRawDataCollection> fedRawDataCollectionToken_;
  };

  TestReadFEDRawDataCollection::TestReadFEDRawDataCollection(edm::ParameterSet const& iPSet)
      : expectedFEDData0_(iPSet.getParameter<std::vector<unsigned int>>("expectedFEDData0")),
        expectedFEDData3_(iPSet.getParameter<std::vector<unsigned int>>("expectedFEDData3")),
        fedRawDataCollectionToken_(consumes(iPSet.getParameter<edm::InputTag>("fedRawDataCollectionTag"))) {}

  void TestReadFEDRawDataCollection::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    auto const& fedRawDataCollection = iEvent.get(fedRawDataCollectionToken_);
    auto const& fedData0 = fedRawDataCollection.FEDData(0);
    if (fedData0.size() != expectedFEDData0_.size()) {
      throwWithMessage("fedData0 does not have expected size");
    }
    for (unsigned int i = 0; i < fedData0.size(); ++i) {
      if (fedData0.data()[i] != expectedFEDData0_[i]) {
        throwWithMessage("fedData0 does not have expected contents");
      }
    }
    auto const& fedData3 = fedRawDataCollection.FEDData(3);
    if (fedData3.size() != expectedFEDData3_.size()) {
      throwWithMessage("fedData3 does not have expected size");
    }
    for (unsigned int i = 0; i < fedData3.size(); ++i) {
      if (fedData3.data()[i] != expectedFEDData3_[i]) {
        throwWithMessage("fedData3 does not have expected contents");
      }
    }
  }

  void TestReadFEDRawDataCollection::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestReadFEDRawDataCollection::analyze, " << msg;
  }

  void TestReadFEDRawDataCollection::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<unsigned int>>("expectedFEDData0");
    desc.add<std::vector<unsigned int>>("expectedFEDData3");
    desc.add<edm::InputTag>("fedRawDataCollectionTag");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestReadFEDRawDataCollection;
DEFINE_FWK_MODULE(TestReadFEDRawDataCollection);
