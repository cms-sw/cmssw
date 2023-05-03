// -*- C++ -*-
//
// Package:    DataFormats/FEDRawData
// Class:      TestWriteFEDRawDataCollection
//
/**\class edmtest::TestWriteFEDRawDataCollection
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

  class TestWriteFEDRawDataCollection : public edm::global::EDProducer<> {
  public:
    TestWriteFEDRawDataCollection(edm::ParameterSet const&);
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    std::vector<unsigned int> fedData0_;
    std::vector<unsigned int> fedData3_;
    edm::EDPutTokenT<FEDRawDataCollection> fedRawDataCollectionPutToken_;
  };

  TestWriteFEDRawDataCollection::TestWriteFEDRawDataCollection(edm::ParameterSet const& iPSet)
      : fedData0_(iPSet.getParameter<std::vector<unsigned int>>("FEDData0")),
        fedData3_(iPSet.getParameter<std::vector<unsigned int>>("FEDData3")),
        fedRawDataCollectionPutToken_(produces()) {}

  void TestWriteFEDRawDataCollection::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    auto fedRawDataCollection = std::make_unique<FEDRawDataCollection>();
    FEDRawData& fedData0 = fedRawDataCollection->FEDData(0);
    FEDRawData& fedData3 = fedRawDataCollection->FEDData(3);

    fedData0.resize(fedData0_.size());
    unsigned char* dataPtr0 = fedData0.data();
    for (unsigned int i = 0; i < fedData0_.size(); ++i) {
      dataPtr0[i] = fedData0_[i];
    }
    fedData3.resize(fedData3_.size());
    unsigned char* dataPtr3 = fedData3.data();
    for (unsigned int i = 0; i < fedData3_.size(); ++i) {
      dataPtr3[i] = fedData3_[i];
    }
    iEvent.put(fedRawDataCollectionPutToken_, std::move(fedRawDataCollection));
  }

  void TestWriteFEDRawDataCollection::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<unsigned int>>("FEDData0");
    desc.add<std::vector<unsigned int>>("FEDData3");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestWriteFEDRawDataCollection;
DEFINE_FWK_MODULE(TestWriteFEDRawDataCollection);
