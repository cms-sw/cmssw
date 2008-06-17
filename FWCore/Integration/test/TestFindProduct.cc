
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

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include <iostream>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edmtest 
{
  class TestFindProduct : public edm::EDAnalyzer
  {
  public:

    explicit TestFindProduct(edm::ParameterSet const& pset);
    virtual ~TestFindProduct();
    
    virtual void analyze(edm::Event const& e, edm::EventSetup const& es);
    virtual void endJob();

  private:

    std::vector<edm::InputTag> inputTags_;
    int expectedSum_;
    int sum_;
  }; // class TestFindProduct

  //--------------------------------------------------------------------
  //
  // Implementation details

  TestFindProduct::TestFindProduct(edm::ParameterSet const& pset) :
    inputTags_(pset.getUntrackedParameter<std::vector<edm::InputTag> >("inputTags")),
    expectedSum_(pset.getUntrackedParameter<int>("expectedSum", 0)),
    sum_(0)
  {
  }

  TestFindProduct::~TestFindProduct() {}

  void
  TestFindProduct::analyze(edm::Event const& e, edm::EventSetup const& es)
  {
    edm::Handle<IntProduct> h;

    for (std::vector<edm::InputTag>::const_iterator iter = inputTags_.begin(),
	   iEnd = inputTags_.end();
         iter != iEnd;
         ++iter) {
      e.getByLabel(*iter, h);
      sum_ += h->value;
    }
  }

  void
  TestFindProduct::endJob()
  {
    std::cout << "TestFindProduct sum = " << sum_ << std::endl;
    if (expectedSum_ != 0 && sum_ != expectedSum_) {
      std::cerr << "TestFindProduct: Sum of test object values does not equal expected value" << std::endl;
      abort();
    }
  }
} // namespace edmtest

using edmtest::TestFindProduct;
DEFINE_FWK_MODULE(TestFindProduct);
