// -*- C++ -*-
//
// Package:    DataFormats/HLTReco
// Class:      TestReadTriggerEvent
//
/**\class edmtest::TestReadTriggerEvent
  Description: Used as part of tests that ensure the trigger::TriggerEvent
  data format can be persistently written and in a subsequent process
  read. First, this is done using the current release version for writing
  and reading. In addition, the output file of the write process should
  be saved permanently each time the trigger::TriggerEvent persistent data
  format changes. In unit tests, we read each of those saved files to verify
  that the current releases can read older versions of the data format.
*/
// Original Author:  W. David Dagenhart
//         Created:  8 May 2023

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
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

#include <string>
#include <vector>

namespace edmtest {

  class TestReadTriggerEvent : public edm::global::EDAnalyzer<> {
  public:
    TestReadTriggerEvent(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    void throwWithMessage(const char*) const;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    // These expected values are meaningless other than we use them
    // to check that values read from persistent storage match the values
    // we know were written.
    std::string expectedUsedProcessName_;
    std::vector<std::string> expectedCollectionTags_;
    std::vector<unsigned int> expectedCollectionKeys_;

    std::vector<int> expectedIds_;
    std::vector<double> expectedPts_;
    std::vector<double> expectedEtas_;
    std::vector<double> expectedPhis_;
    std::vector<double> expectedMasses_;

    std::vector<std::string> expectedFilterTags_;
    unsigned int expectedElementsPerVector_;
    std::vector<int> expectedFilterIds_;
    std::vector<unsigned int> expectedFilterKeys_;

    edm::EDGetTokenT<trigger::TriggerEvent> triggerEventToken_;
  };

  TestReadTriggerEvent::TestReadTriggerEvent(edm::ParameterSet const& iPSet)
      : expectedUsedProcessName_(iPSet.getParameter<std::string>("expectedUsedProcessName")),
        expectedCollectionTags_(iPSet.getParameter<std::vector<std::string>>("expectedCollectionTags")),
        expectedCollectionKeys_(iPSet.getParameter<std::vector<unsigned int>>("expectedCollectionKeys")),
        expectedIds_(iPSet.getParameter<std::vector<int>>("expectedIds")),
        expectedPts_(iPSet.getParameter<std::vector<double>>("expectedPts")),
        expectedEtas_(iPSet.getParameter<std::vector<double>>("expectedEtas")),
        expectedPhis_(iPSet.getParameter<std::vector<double>>("expectedPhis")),
        expectedMasses_(iPSet.getParameter<std::vector<double>>("expectedMasses")),

        expectedFilterTags_(iPSet.getParameter<std::vector<std::string>>("expectedFilterTags")),
        expectedElementsPerVector_(iPSet.getParameter<unsigned int>("expectedElementsPerVector")),
        expectedFilterIds_(iPSet.getParameter<std::vector<int>>("expectedFilterIds")),
        expectedFilterKeys_(iPSet.getParameter<std::vector<unsigned int>>("expectedFilterKeys")),

        triggerEventToken_(consumes(iPSet.getParameter<edm::InputTag>("triggerEventTag"))) {
    if (expectedIds_.size() != expectedPts_.size() || expectedIds_.size() != expectedEtas_.size() ||
        expectedIds_.size() != expectedPhis_.size() || expectedIds_.size() != expectedMasses_.size()) {
      throw cms::Exception("TestFailure")
          << "TestReadTriggerEvent, test configuration error: "
             "expectedIds, expectedPts, expectedEtas, expectedPhis, and expectedMasses should have the same size.";
    }
    if (expectedFilterIds_.size() != expectedElementsPerVector_ * expectedFilterTags_.size() ||
        expectedFilterKeys_.size() != expectedElementsPerVector_ * expectedFilterTags_.size()) {
      throw cms::Exception("TestFailure") << "TestReadTriggerEvent, test configuration error: "
                                             "size of expectedFilterIds and size of expectedFilterKeys "
                                             "should equal size of expectedFilterTags times expectedElementsPerVector";
    }
  }

  void TestReadTriggerEvent::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    auto const& triggerEvent = iEvent.get(triggerEventToken_);

    if (triggerEvent.usedProcessName() != expectedUsedProcessName_) {
      throwWithMessage("usedProcessName does not have expected value");
    }

    if (triggerEvent.collectionTags() != expectedCollectionTags_) {
      throwWithMessage("collectionTags do not have expected values");
    }

    trigger::Keys expectedKeys;
    expectedKeys.reserve(expectedCollectionKeys_.size());
    for (auto const& element : expectedCollectionKeys_) {
      expectedKeys.push_back(static_cast<trigger::size_type>(element));
    }

    if (triggerEvent.collectionKeys() != expectedKeys) {
      throwWithMessage("collectionKeys do not have expected values");
    }

    trigger::TriggerObjectCollection const& triggerObjectCollection = triggerEvent.getObjects();
    if (triggerObjectCollection.size() != expectedIds_.size()) {
      throwWithMessage("triggerObjectCollection does not have expected size");
    }
    for (unsigned int i = 0; i < triggerObjectCollection.size(); ++i) {
      trigger::TriggerObject const& triggerObject = triggerObjectCollection[i];
      if (triggerObject.id() != expectedIds_[i]) {
        throwWithMessage("triggerObjectCollection id does not have expected value");
      }
      if (triggerObject.pt() != static_cast<float>(expectedPts_[i])) {
        throwWithMessage("triggerObjectCollection pt does not have expected value");
      }
      if (triggerObject.eta() != static_cast<float>(expectedEtas_[i])) {
        throwWithMessage("triggerObjectCollection eta does not have expected value");
      }
      if (triggerObject.phi() != static_cast<float>(expectedPhis_[i])) {
        throwWithMessage("triggerObjectCollection phi does not have expected value");
      }
      if (triggerObject.mass() != static_cast<float>(expectedMasses_[i])) {
        throwWithMessage("triggerObjectCollection mass does not have expected value");
      }
    }

    if (triggerEvent.sizeFilters() != expectedFilterTags_.size()) {
      throwWithMessage("triggerFilters does not have expected size");
    }

    for (unsigned int i = 0; i < expectedFilterTags_.size(); ++i) {
      if (triggerEvent.filterLabel(i) != expectedFilterTags_[i]) {
        throwWithMessage("filterTags does not have expected value");
      }
      trigger::Vids const& filterIds = triggerEvent.filterIds(i);
      if (filterIds.size() != expectedElementsPerVector_) {
        throwWithMessage("filterIds does not have expected size");
      }
      trigger::Keys const& filterKeys = triggerEvent.filterKeys(i);
      if (filterKeys.size() != expectedElementsPerVector_) {
        throwWithMessage("filterKeys does not have expected size");
      }
      for (unsigned int j = 0; j < expectedElementsPerVector_; ++j) {
        if (filterIds[j] != expectedFilterIds_[i * expectedElementsPerVector_ + j]) {
          throwWithMessage("filterIds does not contain expected values");
        }
        if (filterKeys[j] != expectedFilterKeys_[i * expectedElementsPerVector_ + j]) {
          throwWithMessage("filterKeys does not contain expected values");
        }
      }
    }
  }

  void TestReadTriggerEvent::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestReadTriggerEvent::analyze, " << msg;
  }

  void TestReadTriggerEvent::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("expectedUsedProcessName");
    desc.add<std::vector<std::string>>("expectedCollectionTags");
    desc.add<std::vector<unsigned int>>("expectedCollectionKeys");
    desc.add<std::vector<int>>("expectedIds");
    desc.add<std::vector<double>>("expectedPts");
    desc.add<std::vector<double>>("expectedEtas");
    desc.add<std::vector<double>>("expectedPhis");
    desc.add<std::vector<double>>("expectedMasses");
    desc.add<std::vector<std::string>>("expectedFilterTags");
    desc.add<unsigned int>("expectedElementsPerVector");
    desc.add<std::vector<int>>("expectedFilterIds");
    desc.add<std::vector<unsigned int>>("expectedFilterKeys");
    desc.add<edm::InputTag>("triggerEventTag");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestReadTriggerEvent;
DEFINE_FWK_MODULE(TestReadTriggerEvent);
