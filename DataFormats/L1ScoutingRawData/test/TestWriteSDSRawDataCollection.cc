#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
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

  class TestWriteSDSRawDataCollection : public edm::global::EDProducer<> {
  public:
    TestWriteSDSRawDataCollection(edm::ParameterSet const&);
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    std::vector<unsigned int> sdsData1_;
    std::vector<unsigned int> sdsData2_;
    edm::EDPutTokenT<SDSRawDataCollection> sdsRawDataCollectionPutToken_;
  };

  TestWriteSDSRawDataCollection::TestWriteSDSRawDataCollection(edm::ParameterSet const& iPSet)
      : sdsData1_(iPSet.getParameter<std::vector<unsigned int>>("SDSData1")),
        sdsData2_(iPSet.getParameter<std::vector<unsigned int>>("SDSData2")),
        sdsRawDataCollectionPutToken_(produces()) {}

  void TestWriteSDSRawDataCollection::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    auto sdsRawDataCollection = std::make_unique<SDSRawDataCollection>();
    FEDRawData& fedData1 = sdsRawDataCollection->FEDData(1);
    FEDRawData& fedData2 = sdsRawDataCollection->FEDData(2);

    fedData1.resize(sdsData1_.size(), 4);
    unsigned char* dataPtr1 = fedData1.data();
    for (unsigned int i = 0; i < sdsData1_.size(); ++i) {
      dataPtr1[i] = sdsData1_[i];
    }
    fedData2.resize(sdsData2_.size(), 4);
    unsigned char* dataPtr2 = fedData2.data();
    for (unsigned int i = 0; i < sdsData2_.size(); ++i) {
      dataPtr2[i] = sdsData2_[i];
    }
    iEvent.put(sdsRawDataCollectionPutToken_, std::move(sdsRawDataCollection));
  }

  void TestWriteSDSRawDataCollection::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<unsigned int>>("SDSData1");
    desc.add<std::vector<unsigned int>>("SDSData2");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestWriteSDSRawDataCollection;
DEFINE_FWK_MODULE(TestWriteSDSRawDataCollection);