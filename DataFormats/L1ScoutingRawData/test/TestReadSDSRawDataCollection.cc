#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
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

  class TestReadSDSRawDataCollection : public edm::global::EDAnalyzer<> {
  public:
    TestReadSDSRawDataCollection(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    void throwWithMessage(const char*) const;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    std::vector<unsigned int> expectedSDSData1_;
    std::vector<unsigned int> expectedSDSData2_;
    edm::EDGetTokenT<SDSRawDataCollection> sdsRawDataCollectionToken_;
  };

  TestReadSDSRawDataCollection::TestReadSDSRawDataCollection(edm::ParameterSet const& iPSet)
      : expectedSDSData1_(iPSet.getParameter<std::vector<unsigned int>>("expectedSDSData1")),
        expectedSDSData2_(iPSet.getParameter<std::vector<unsigned int>>("expectedSDSData2")),
        sdsRawDataCollectionToken_(consumes(iPSet.getParameter<edm::InputTag>("sdsRawDataCollectionTag"))) {}

  void TestReadSDSRawDataCollection::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    auto const& sdsRawDataCollection = iEvent.get(sdsRawDataCollectionToken_);
    auto const& sdsData1 = sdsRawDataCollection.FEDData(1);
    if (sdsData1.size() != expectedSDSData1_.size()) {
      throwWithMessage("sdsData1 does not have expected size");
    }
    for (unsigned int i = 0; i < sdsData1.size(); ++i) {
      if (sdsData1.data()[i] != expectedSDSData1_[i]) {
        throwWithMessage("sdsData1 does not have expected contents");
      }
    }
    auto const& sdsData2 = sdsRawDataCollection.FEDData(2);
    if (sdsData2.size() != expectedSDSData2_.size()) {
      throwWithMessage("sdsData2 does not have expected size");
    }
    for (unsigned int i = 0; i < sdsData2.size(); ++i) {
      if (sdsData2.data()[i] != expectedSDSData2_[i]) {
        throwWithMessage("sdsData2 does not have expected contents");
      }
    }
  }

  void TestReadSDSRawDataCollection::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestReadSDSRawDataCollection::analyze, " << msg;
  }

  void TestReadSDSRawDataCollection::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<unsigned int>>("expectedSDSData1");
    desc.add<std::vector<unsigned int>>("expectedSDSData2");
    desc.add<edm::InputTag>("sdsRawDataCollectionTag");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestReadSDSRawDataCollection;
DEFINE_FWK_MODULE(TestReadSDSRawDataCollection);