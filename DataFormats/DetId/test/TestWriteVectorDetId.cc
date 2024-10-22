// -*- C++ -*-
//
// Package:    DataFormats/DetId
// Class:      TestWriteVectorDetId
//
/**\class edmtest::TestWriteVectorDetId
  Description: Used as part of tests that ensure the std::vector<DetId> raw
  data format type can be persistently written and in a subsequent process
  read. First, this is done using the current release version for writing
  and reading. In addition, the output file of the write process should
  be saved permanently each time the std::vector<DetId> persistent data
  format changes. In unit tests, we read each of those saved files to verify
  that the current releases can read older versions of the data format.
*/
// Original Author:  W. David Dagenhart
//         Created:  25 Sep 2023

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <memory>
#include <utility>
#include <vector>

namespace edmtest {

  class TestWriteVectorDetId : public edm::global::EDProducer<> {
  public:
    TestWriteVectorDetId(edm::ParameterSet const&);
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    unsigned int testValue_;
    edm::EDPutTokenT<std::vector<DetId>> putToken_;
  };

  TestWriteVectorDetId::TestWriteVectorDetId(edm::ParameterSet const& iPSet)
      : testValue_(iPSet.getParameter<unsigned int>("testValue")), putToken_(produces()) {}

  void TestWriteVectorDetId::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    // Fill a std::vector<DetId>. Make sure it is not empty.
    // We will test in a later process that we read the same
    // value we wrote, but the value is otherwise meaningless.
    // The only purpose is to test that ROOT can read the bits
    // properly (maybe years and many ROOT and CSSSW versions
    // after the files were written).

    auto vectorDetIds = std::make_unique<std::vector<DetId>>();

    unsigned int numberDetIds = (iEvent.id().event() - 1) % 10;
    unsigned int detId = testValue_ + iEvent.id().event();
    for (unsigned int i = 0; i < numberDetIds; ++i) {
      detId += iEvent.id().event();
      vectorDetIds->emplace_back(detId);
    }
    iEvent.put(putToken_, std::move(vectorDetIds));
  }

  void TestWriteVectorDetId::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<unsigned int>("testValue");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestWriteVectorDetId;
DEFINE_FWK_MODULE(TestWriteVectorDetId);
