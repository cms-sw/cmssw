
// This test module will look for IntProducts in Events.
// The number of IntProducts and their InputTags (label,
// instance, process) must be configured.

// One can also configure an expected value for the sum of
// all the values in the IntProducts that are found.  Note
// that an IntProduct is just a test product that simply
// contains a single integer.

// If the products are not found, then an exception is thrown.
// If the sum does not match there is an error message and
// an abort.

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/getProducerParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include <iostream>
#include <vector>

namespace edmtest {
  class TestFindProduct : public edm::EDAnalyzer {
  public:

    explicit TestFindProduct(edm::ParameterSet const& pset);
    virtual ~TestFindProduct();

    virtual void analyze(edm::Event const& e, edm::EventSetup const& es);
    virtual void endJob();

  private:

    std::vector<edm::InputTag> inputTags_;
    int expectedSum_;
    int sum_;
    std::vector<edm::InputTag> inputTagsNotFound_;
    bool getByTokenFirst_;
    std::vector<edm::InputTag> inputTagsView_;
    bool runProducerParameterCheck_;

    std::vector<edm::EDGetTokenT<IntProduct> > tokens_;
    std::vector<edm::EDGetTokenT<IntProduct> > tokensNotFound_;
    std::vector<edm::EDGetTokenT<edm::View<int> > > tokensView_;
  }; // class TestFindProduct

  //--------------------------------------------------------------------
  //
  // Implementation details

  TestFindProduct::TestFindProduct(edm::ParameterSet const& pset) :
    inputTags_(pset.getUntrackedParameter<std::vector<edm::InputTag> >("inputTags")),
    expectedSum_(pset.getUntrackedParameter<int>("expectedSum", 0)),
    sum_(0),
    inputTagsNotFound_(),
    getByTokenFirst_(pset.getUntrackedParameter<bool>("getByTokenFirst", false)),
    inputTagsView_(),
    runProducerParameterCheck_(pset.getUntrackedParameter<bool>("runProducerParameterCheck", false))
  {
    std::vector<edm::InputTag> emptyTagVector;
    inputTagsNotFound_ = pset.getUntrackedParameter<std::vector<edm::InputTag> >("inputTagsNotFound", emptyTagVector);
    inputTagsView_ = pset.getUntrackedParameter<std::vector<edm::InputTag> >("inputTagsView", emptyTagVector);

    for (auto const& tag : inputTags_) {
      tokens_.push_back(consumes<IntProduct>(tag));
    }
    for (auto const& tag : inputTagsNotFound_) {
      tokensNotFound_.push_back(consumes<IntProduct>(tag));
    }
    for (auto const& tag : inputTagsView_) {
      tokensView_.push_back(consumes<edm::View<int> >(tag));
    }
  }

  TestFindProduct::~TestFindProduct() {}

  void
  TestFindProduct::analyze(edm::Event const& e, edm::EventSetup const&) {

    edm::Handle<IntProduct> h;
    edm::Handle<IntProduct> hToken;
    edm::Handle<edm::View<int> > hView;
    edm::Handle<edm::View<int> > hViewToken;

    std::vector<edm::EDGetTokenT<IntProduct> >::const_iterator iToken = tokens_.begin();
    for(std::vector<edm::InputTag>::const_iterator iter = inputTags_.begin(),
         iEnd = inputTags_.end();
         iter != iEnd;
        ++iter, ++iToken) {

      if(getByTokenFirst_) {
        e.getByToken(*iToken, hToken);
        *hToken;
      }

      e.getByLabel(*iter, h);
      sum_ += h->value;

      e.getByToken(*iToken, hToken);
      if (h->value != hToken->value) {
        std::cerr << "TestFindProduct::analyze getByLabel and getByToken return inconsistent results " << std::endl;
        abort();    
      }

      if (runProducerParameterCheck_) {
        edm::ParameterSet const* producerPset = edm::getProducerParameterSet(*hToken.provenance());
        int par = producerPset->getParameter<int>("ivalue");
        // These expected values are just from knowing the values in the
        // configuration files for this test.
        int expectedParameterValue = 3;
        if (!iter->process().empty()) {
          if (e.run() == 1) {
            expectedParameterValue = 1;
          } else {
            expectedParameterValue = 2;
          }
        }
        if (par != expectedParameterValue) {
          std::cerr << "TestFindProduct::analyze unexpected value from producer parameter set" << std::endl;
          abort();
        }
      }
    }
    iToken = tokensNotFound_.begin();
    for(std::vector<edm::InputTag>::const_iterator iter = inputTagsNotFound_.begin(),
         iEnd = inputTagsNotFound_.end();
         iter != iEnd;
        ++iter, ++iToken) {
      e.getByLabel(*iter, h);
      if (h.isValid()) {
        std::cerr << "TestFindProduct::analyze: getByLabel found a product that should not be found "
                  << h.provenance()->moduleLabel() << std::endl;
        abort();    
      }

      e.getByToken(*iToken, hToken);
      if (hToken.isValid()) {
        std::cerr << "TestFindProduct::analyze: getByToken found a product that should not be found "
                  << hToken.provenance()->moduleLabel() << std::endl;
        abort();    
      }
    }
    std::vector<edm::EDGetTokenT<edm::View<int> > >::const_iterator iTokenView = tokensView_.begin();
    for(std::vector<edm::InputTag>::const_iterator iter = inputTagsView_.begin(),
         iEnd = inputTagsView_.end();
         iter != iEnd;
        ++iter, ++iTokenView) {

      if(getByTokenFirst_) {
        e.getByToken(*iTokenView, hViewToken);
        *hViewToken;
      }

      e.getByLabel(*iter, hView);
      sum_ += hView->at(0);

      e.getByToken(*iTokenView, hViewToken);
      if (hView->at(0) != hViewToken->at(0)) {
        std::cerr << "TestFindProduct::analyze getByLabel and getByToken return inconsistent results " << std::endl;
        abort();    
      }
    }
  }

  void
  TestFindProduct::endJob() {
    std::cout << "TestFindProduct sum = " << sum_ << std::endl;
    if(expectedSum_ != 0 && sum_ != expectedSum_) {
      throw cms::Exception("TestFail")
        << "TestFindProduct::endJob - Sum of test object values does not equal expected value";
    }
  }
} // namespace edmtest

using edmtest::TestFindProduct;
DEFINE_FWK_MODULE(TestFindProduct);
