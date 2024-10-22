// -*- C++ -*-
//
// Package:    DataFormats/HLTReco
// Class:      TestWriteTriggerEvent
//
/**\class edmtest::TestWriteTriggerEvent
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
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace edmtest {

  class TestWriteTriggerEvent : public edm::global::EDProducer<> {
  public:
    TestWriteTriggerEvent(edm::ParameterSet const&);
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    std::string usedProcessName_;
    std::vector<std::string> collectionTags_;
    std::vector<unsigned int> collectionKeys_;
    std::vector<int> ids_;
    std::vector<double> pts_;
    std::vector<double> etas_;
    std::vector<double> phis_;
    std::vector<double> masses_;
    std::vector<std::string> filterTags_;
    unsigned int elementsPerVector_;
    std::vector<int> filterIds_;
    std::vector<unsigned int> filterKeys_;

    edm::EDPutTokenT<trigger::TriggerEvent> triggerEventPutToken_;
  };

  TestWriteTriggerEvent::TestWriteTriggerEvent(edm::ParameterSet const& iPSet)
      : usedProcessName_(iPSet.getParameter<std::string>("usedProcessName")),
        collectionTags_(iPSet.getParameter<std::vector<std::string>>("collectionTags")),
        collectionKeys_(iPSet.getParameter<std::vector<unsigned int>>("collectionKeys")),
        ids_(iPSet.getParameter<std::vector<int>>("ids")),
        pts_(iPSet.getParameter<std::vector<double>>("pts")),
        etas_(iPSet.getParameter<std::vector<double>>("etas")),
        phis_(iPSet.getParameter<std::vector<double>>("phis")),
        masses_(iPSet.getParameter<std::vector<double>>("masses")),
        filterTags_(iPSet.getParameter<std::vector<std::string>>("filterTags")),
        elementsPerVector_(iPSet.getParameter<unsigned int>("elementsPerVector")),
        filterIds_(iPSet.getParameter<std::vector<int>>("filterIds")),
        filterKeys_(iPSet.getParameter<std::vector<unsigned int>>("filterKeys")),

        triggerEventPutToken_(produces()) {
    if (ids_.size() != pts_.size() || ids_.size() != etas_.size() || ids_.size() != phis_.size() ||
        ids_.size() != masses_.size()) {
      throw cms::Exception("TestFailure") << "TestWriteTriggerEvent, test configuration error: "
                                             "ids, pts, etas, phis, and masses should have the same size.";
    }
    if (filterIds_.size() != elementsPerVector_ * filterTags_.size() ||
        filterKeys_.size() != elementsPerVector_ * filterTags_.size()) {
      throw cms::Exception("TestFailure")
          << "TestWriteTriggerEvent, test configuration error: "
             "size of filterIds and size of filterKeys should equal size of filterTags times elementsPerVector";
    }
  }

  void TestWriteTriggerEvent::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    // Fill a TriggerEvent object. Make sure all the containers inside
    // of it have something in them (not empty). The values are meaningless.
    // We will later check that after writing this object to persistent storage
    // and then reading it in a later process we obtain matching values for
    // all this content.

    auto triggerEvent = std::make_unique<trigger::TriggerEvent>(
        usedProcessName_, collectionTags_.size(), ids_.size(), filterTags_.size());
    trigger::Keys keys;
    keys.reserve(collectionKeys_.size());
    for (auto const& element : collectionKeys_) {
      keys.push_back(static_cast<trigger::size_type>(element));
    }
    triggerEvent->addCollections(collectionTags_, keys);

    trigger::TriggerObjectCollection triggerObjectCollection;
    triggerObjectCollection.reserve(ids_.size());
    for (unsigned int i = 0; i < ids_.size(); ++i) {
      triggerObjectCollection.emplace_back(ids_[i],
                                           static_cast<float>(pts_[i]),
                                           static_cast<float>(etas_[i]),
                                           static_cast<float>(phis_[i]),
                                           static_cast<float>(masses_[i]));
    }
    triggerEvent->addObjects(triggerObjectCollection);

    for (unsigned int i = 0; i < filterTags_.size(); ++i) {
      trigger::Vids filterIds;
      filterIds.reserve(elementsPerVector_);
      trigger::Keys filterKeys;
      filterKeys.reserve(elementsPerVector_);
      for (unsigned int j = 0; j < elementsPerVector_; ++j) {
        filterIds.push_back(filterIds_[i * elementsPerVector_ + j]);
        filterKeys.push_back(filterKeys_[i * elementsPerVector_ + j]);
      }
      triggerEvent->addFilter(edm::InputTag(filterTags_[i]), filterIds, filterKeys);
    }

    iEvent.put(triggerEventPutToken_, std::move(triggerEvent));
  }

  void TestWriteTriggerEvent::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("usedProcessName");
    desc.add<std::vector<std::string>>("collectionTags");
    desc.add<std::vector<unsigned int>>("collectionKeys");
    desc.add<std::vector<int>>("ids");
    desc.add<std::vector<double>>("pts");
    desc.add<std::vector<double>>("etas");
    desc.add<std::vector<double>>("phis");
    desc.add<std::vector<double>>("masses");
    desc.add<std::vector<std::string>>("filterTags");
    desc.add<unsigned int>("elementsPerVector");
    desc.add<std::vector<int>>("filterIds");
    desc.add<std::vector<unsigned int>>("filterKeys");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestWriteTriggerEvent;
DEFINE_FWK_MODULE(TestWriteTriggerEvent);
