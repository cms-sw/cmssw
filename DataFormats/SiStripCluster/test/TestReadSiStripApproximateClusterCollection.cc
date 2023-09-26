// -*- C++ -*-
//
// Package:    DataFormats/SiStripCluster
// Class:      TestReadSiStripApproximateClusterCollection
//
/**\class edmtest::TestReadSiStripApproximateClusterCollection
  Description: Used as part of tests that ensure the SiStripApproximateClusterCollection
  data format can be persistently written and in a subsequent process
  read. First, this is done using the current release version for writing
  and reading. In addition, the output file of the write process should
  be saved permanently each time the SiStripApproximateClusterCollection persistent data
  format changes. In unit tests, we read each of those saved files to verify
  that the current releases can read older versions of the data format.
*/
// Original Author:  W. David Dagenhart
//         Created:  22 September 2023

#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollection.h"
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

  class TestReadSiStripApproximateClusterCollection : public edm::global::EDAnalyzer<> {
  public:
    TestReadSiStripApproximateClusterCollection(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    void throwWithMessage(const char*) const;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    // These expected values are meaningless other than we use them
    // to check that values read from persistent storage match the values
    // we know were written.
    std::vector<unsigned int> expectedIntegralValues_;

    edm::EDGetTokenT<SiStripApproximateClusterCollection> collectionToken_;
  };

  TestReadSiStripApproximateClusterCollection::TestReadSiStripApproximateClusterCollection(
      edm::ParameterSet const& iPSet)
      : expectedIntegralValues_(iPSet.getParameter<std::vector<unsigned int>>("expectedIntegralValues")),
        collectionToken_(consumes(iPSet.getParameter<edm::InputTag>("collectionTag"))) {
    if (expectedIntegralValues_.size() != 7) {
      throw cms::Exception("TestFailure") << "TestReadSiStripApproximateClusterCollection, test configuration error: "
                                             "expectedIntegralValues should have size 7.";
    }
  }

  void TestReadSiStripApproximateClusterCollection::analyze(edm::StreamID,
                                                            edm::Event const& iEvent,
                                                            edm::EventSetup const&) const {
    auto const& siStripApproximateClusterCollection = iEvent.get(collectionToken_);

    unsigned int expectedNumberOfDetIds = (iEvent.id().event() - 1) % 10;
    unsigned int expectedDetId = expectedIntegralValues_[0] + iEvent.id().event();
    unsigned int numberOfDetIds = 0;
    for (const auto& detClusters : siStripApproximateClusterCollection) {
      ++numberOfDetIds;
      expectedDetId += iEvent.id().event();
      if (detClusters.id() != expectedDetId) {
        throwWithMessage("DetId in detClusters does not have expected value");
      }
      unsigned int expectedNumberOfClustersPerDetId = (iEvent.id().event() - 1) % 10;
      unsigned int j = 0;
      for (const auto& cluster : detClusters) {
        unsigned int iOffset = j + iEvent.id().event();
        if (cluster.barycenter() != expectedIntegralValues_[1] + iOffset) {
          throwWithMessage("barycenter does not have expected value");
        }
        if (cluster.width() != expectedIntegralValues_[2] + iOffset) {
          throwWithMessage("width does not have expected value");
        }
        if (cluster.avgCharge() != expectedIntegralValues_[3] + iOffset) {
          throwWithMessage("avgCharge does not have expected value");
        }
        if (cluster.filter() != (j < (expectedIntegralValues_[4] + iEvent.id().event()) % 10)) {
          throwWithMessage("filter does not have expected value");
        }
        if (cluster.isSaturated() != (j < (expectedIntegralValues_[5] + iEvent.id().event()) % 10)) {
          throwWithMessage("isSaturated does not have expected value");
        }
        if (cluster.peakFilter() != (j < (expectedIntegralValues_[6] + iEvent.id().event()) % 10)) {
          throwWithMessage("peakFilter does not have expected value");
        }
        ++j;
      }
      if (j != expectedNumberOfClustersPerDetId) {
        throwWithMessage("Number of cluster for DetId does not have expected value");
      }
    }
    if (numberOfDetIds != expectedNumberOfDetIds) {
      throwWithMessage("Number of DetIds does not match expected value");
    }
  }

  void TestReadSiStripApproximateClusterCollection::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestReadSiStripApproximateClusterCollection::analyze, " << msg;
  }

  void TestReadSiStripApproximateClusterCollection::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<unsigned int>>("expectedIntegralValues");
    desc.add<edm::InputTag>("collectionTag");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestReadSiStripApproximateClusterCollection;
DEFINE_FWK_MODULE(TestReadSiStripApproximateClusterCollection);
