// -*- C++ -*-
//
// Package:    DataFormats/SiStripCluster
// Class:      TestWriteSiStripApproximateClusterCollection
//
/**\class edmtest::TestWriteSiStripApproximateClusterCollection
  Description: Used as part of tests that ensure the SiStripApproximateClusterCollection
  data format can be persistently written and in a subsequent process
  read. First, this is done using the current release version for writing
  and reading. In addition, the output file of the write process should
  be saved permanently each time the SiStripApproximateClusterCollection persistent data
  format changes. In unit tests, we read each of those saved files to verify
  that the current releases can read older versions of the data format.
*/
// Original Author:  W. David Dagenhart
//         Created:  20 Sep 2023

#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollection.h"
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
#include <utility>
#include <vector>

namespace edmtest {

  class TestWriteSiStripApproximateClusterCollection : public edm::global::EDProducer<> {
  public:
    TestWriteSiStripApproximateClusterCollection(edm::ParameterSet const&);
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    std::vector<unsigned int> integralValues_;
    edm::EDPutTokenT<SiStripApproximateClusterCollection> collectionPutToken_;
  };

  TestWriteSiStripApproximateClusterCollection::TestWriteSiStripApproximateClusterCollection(
      edm::ParameterSet const& iPSet)
      : integralValues_(iPSet.getParameter<std::vector<unsigned int>>("integralValues")),
        collectionPutToken_(produces()) {
    if (integralValues_.size() != 7) {
      throw cms::Exception("TestFailure") << "TestWriteSiStripApproximateClusterCollection, test configuration error: "
                                             "integralValues should have size 7 and it doesn't";
    }
  }

  void TestWriteSiStripApproximateClusterCollection::produce(edm::StreamID,
                                                             edm::Event& iEvent,
                                                             edm::EventSetup const&) const {
    // Fill a SiStripApproximateClusterCollection. Make sure all the containers inside
    // of it have something in them (not empty). There is a little complexity here
    // to vary the values written and the sizes of the containers, but the values and
    // sizes are meaningless other than we want test patterns that aren't all the same
    // value and same size for a better test.
    // We will later check that after writing this object to persistent storage
    // and then reading it in a later process we obtain matching values for
    // all this content. I imagine if you tried to run reconstruction on these
    // objects it would crash badly. Here, the only purpose is to test that ROOT
    // can read the bits properly (maybe years and many ROOT and CSSSW versions
    // after the files were written).

    auto siStripApproximateClusterCollection = std::make_unique<SiStripApproximateClusterCollection>();

    unsigned int numberDetIds = (iEvent.id().event() - 1) % 10;
    unsigned int detId = integralValues_[0] + iEvent.id().event();
    for (unsigned int i = 0; i < numberDetIds; ++i) {
      detId += iEvent.id().event();
      auto filler = siStripApproximateClusterCollection->beginDet(detId);
      unsigned int numberOfClustersPerDetId = (iEvent.id().event() - 1) % 10;
      for (unsigned int j = 0; j < numberOfClustersPerDetId; ++j) {
        unsigned int iOffset = j + iEvent.id().event();
        cms_uint16_t barycenter = integralValues_[1] + iOffset;
        cms_uint8_t width = integralValues_[2] + iOffset;
        cms_uint8_t avgCharge = integralValues_[3] + iOffset;
        bool filter = j < (integralValues_[4] + iEvent.id().event()) % 10;
        bool isSaturated = j < (integralValues_[5] + iEvent.id().event()) % 10;
        bool peakFilter = j < (integralValues_[6] + iEvent.id().event()) % 10;
        SiStripApproximateCluster cluster(barycenter, width, avgCharge, filter, isSaturated, peakFilter);
        filler.push_back(cluster);
      }
    }
    iEvent.put(collectionPutToken_, std::move(siStripApproximateClusterCollection));
  }

  void TestWriteSiStripApproximateClusterCollection::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<unsigned int>>("integralValues");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestWriteSiStripApproximateClusterCollection;
DEFINE_FWK_MODULE(TestWriteSiStripApproximateClusterCollection);
