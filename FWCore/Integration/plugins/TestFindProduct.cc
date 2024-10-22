
// This test module will try to get IntProducts in Events,
// Lumis, Runs and ProcessBlocks. The number of IntProducts
// and their InputTags (label, instance, process) must be configured.

// One can also configure an expected value for the sum of
// all the values in the IntProducts that are found.  Note
// that an IntProduct is just a test product that simply
// contains a single integer.

// If the expected products are not found or some other
// unexpected behavior occurs, an exception will be thrown.

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/CacheHandle.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/getProducerParameterSet.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/moduleAbilities.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/ProcessMatch.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <functional>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

namespace edmtest {

  class TestFindProduct : public edm::one::EDAnalyzer<edm::one::WatchRuns,
                                                      edm::one::WatchLuminosityBlocks,
                                                      edm::WatchProcessBlock,
                                                      edm::InputProcessBlockCache<int, long long int>> {
  public:
    explicit TestFindProduct(edm::ParameterSet const& pset);
    ~TestFindProduct() override;

    void analyze(edm::Event const& e, edm::EventSetup const& es) override;
    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void endRun(edm::Run const&, edm::EventSetup const&) override;
    void beginProcessBlock(edm::ProcessBlock const&) override;
    void accessInputProcessBlock(edm::ProcessBlock const&) override;
    void endProcessBlock(edm::ProcessBlock const&) override;
    void endJob() override;

  private:
    std::vector<edm::InputTag> inputTags_;
    int expectedSum_;
    int expectedCache_;
    int sum_;
    std::vector<edm::InputTag> inputTagsNotFound_;
    bool getByTokenFirst_;
    std::vector<edm::InputTag> inputTagsView_;
    bool runProducerParameterCheck_;
    bool testGetterOfProducts_;

    std::vector<edm::InputTag> inputTagsUInt64_;
    std::vector<edm::InputTag> inputTagsEndLumi_;
    std::vector<edm::InputTag> inputTagsEndRun_;
    std::vector<edm::InputTag> inputTagsBeginProcessBlock_;
    std::vector<edm::InputTag> inputTagsInputProcessBlock_;
    std::vector<edm::InputTag> inputTagsEndProcessBlock_;
    std::vector<edm::InputTag> inputTagsEndProcessBlock2_;
    std::vector<edm::InputTag> inputTagsEndProcessBlock3_;
    std::vector<edm::InputTag> inputTagsEndProcessBlock4_;

    std::vector<edm::EDGetTokenT<IntProduct>> tokens_;
    std::vector<edm::EDGetTokenT<IntProduct>> tokensNotFound_;
    std::vector<edm::EDGetTokenT<edm::View<int>>> tokensView_;
    std::vector<edm::EDGetTokenT<UInt64Product>> tokensUInt64_;
    std::vector<edm::EDGetTokenT<IntProduct>> tokensEndLumi_;
    std::vector<edm::EDGetTokenT<IntProduct>> tokensEndRun_;
    std::vector<edm::EDGetTokenT<IntProduct>> tokensBeginProcessBlock_;
    std::vector<edm::EDGetTokenT<IntProduct>> tokensInputProcessBlock_;
    std::vector<edm::EDGetTokenT<IntProduct>> tokensEndProcessBlock_;
    std::vector<edm::EDGetToken> tokensEndProcessBlock2_;
    std::vector<edm::EDGetTokenT<IntProduct>> tokensEndProcessBlock3_;
    std::vector<edm::EDGetTokenT<IntProduct>> tokensEndProcessBlock4_;

    edm::GetterOfProducts<IntProduct> getterOfProducts_;

  };  // class TestFindProduct

  //--------------------------------------------------------------------
  //
  // Implementation details

  TestFindProduct::TestFindProduct(edm::ParameterSet const& pset)
      : inputTags_(pset.getUntrackedParameter<std::vector<edm::InputTag>>("inputTags")),
        expectedSum_(pset.getUntrackedParameter<int>("expectedSum", 0)),
        expectedCache_(pset.getUntrackedParameter<int>("expectedCache", 0)),
        sum_(0),
        inputTagsNotFound_(),
        getByTokenFirst_(pset.getUntrackedParameter<bool>("getByTokenFirst", false)),
        inputTagsView_(),
        runProducerParameterCheck_(pset.getUntrackedParameter<bool>("runProducerParameterCheck", false)),
        testGetterOfProducts_(pset.getUntrackedParameter<bool>("testGetterOfProducts", false)),
        getterOfProducts_(edm::ProcessMatch("*"), this, edm::InProcess) {
    if (testGetterOfProducts_) {
      callWhenNewProductsRegistered(getterOfProducts_);
    }
    std::vector<edm::InputTag> emptyTagVector;
    inputTagsNotFound_ = pset.getUntrackedParameter<std::vector<edm::InputTag>>("inputTagsNotFound", emptyTagVector);
    inputTagsView_ = pset.getUntrackedParameter<std::vector<edm::InputTag>>("inputTagsView", emptyTagVector);
    inputTagsUInt64_ = pset.getUntrackedParameter<std::vector<edm::InputTag>>("inputTagsUInt64", emptyTagVector);
    inputTagsEndLumi_ = pset.getUntrackedParameter<std::vector<edm::InputTag>>("inputTagsEndLumi", emptyTagVector);
    inputTagsEndRun_ = pset.getUntrackedParameter<std::vector<edm::InputTag>>("inputTagsEndRun", emptyTagVector);
    inputTagsBeginProcessBlock_ =
        pset.getUntrackedParameter<std::vector<edm::InputTag>>("inputTagsBeginProcessBlock", emptyTagVector);
    inputTagsInputProcessBlock_ =
        pset.getUntrackedParameter<std::vector<edm::InputTag>>("inputTagsInputProcessBlock", emptyTagVector);
    inputTagsEndProcessBlock_ =
        pset.getUntrackedParameter<std::vector<edm::InputTag>>("inputTagsEndProcessBlock", emptyTagVector);
    inputTagsEndProcessBlock2_ =
        pset.getUntrackedParameter<std::vector<edm::InputTag>>("inputTagsEndProcessBlock2", emptyTagVector);
    inputTagsEndProcessBlock3_ =
        pset.getUntrackedParameter<std::vector<edm::InputTag>>("inputTagsEndProcessBlock3", emptyTagVector);
    inputTagsEndProcessBlock4_ =
        pset.getUntrackedParameter<std::vector<edm::InputTag>>("inputTagsEndProcessBlock4", emptyTagVector);

    for (auto const& tag : inputTags_) {
      tokens_.push_back(consumes<IntProduct>(tag));
    }
    for (auto const& tag : inputTagsNotFound_) {
      tokensNotFound_.push_back(consumes<IntProduct>(tag));
    }
    for (auto const& tag : inputTagsView_) {
      tokensView_.push_back(consumes<edm::View<int>>(tag));
    }
    for (auto const& tag : inputTagsUInt64_) {
      tokensUInt64_.push_back(consumes<UInt64Product>(tag));
    }
    for (auto const& tag : inputTagsEndLumi_) {
      tokensEndLumi_.push_back(consumes<IntProduct, edm::InLumi>(tag));
    }
    for (auto const& tag : inputTagsEndRun_) {
      tokensEndRun_.push_back(consumes<IntProduct, edm::InRun>(tag));
    }
    for (auto const& tag : inputTagsBeginProcessBlock_) {
      tokensBeginProcessBlock_.push_back(consumes<IntProduct, edm::InProcess>(tag));
    }
    for (auto const& tag : inputTagsInputProcessBlock_) {
      tokensInputProcessBlock_.push_back(consumes<IntProduct, edm::InProcess>(tag));
    }
    for (auto const& tag : inputTagsEndProcessBlock_) {
      tokensEndProcessBlock_.push_back(consumes<IntProduct, edm::InProcess>(tag));
    }
    for (auto const& tag : inputTagsEndProcessBlock2_) {
      tokensEndProcessBlock2_.push_back(consumes<IntProduct, edm::InProcess>(tag));
    }
    for (auto const& tag : inputTagsEndProcessBlock3_) {
      tokensEndProcessBlock3_.push_back(consumes<IntProduct, edm::InProcess>(tag));
    }
    for (auto const& tag : inputTagsEndProcessBlock4_) {
      tokensEndProcessBlock4_.push_back(consumes<IntProduct, edm::InProcess>(tag));
    }

    if (!tokensInputProcessBlock_.empty()) {
      registerProcessBlockCacheFiller<int>(
          tokensInputProcessBlock_[0],
          [this](edm::ProcessBlock const& processBlock, std::shared_ptr<int> const& previousCache) {
            auto returnValue = std::make_shared<int>(0);
            for (auto const& token : tokensInputProcessBlock_) {
              *returnValue += processBlock.get(token).value;
            }
            return returnValue;
          });
      registerProcessBlockCacheFiller<1>(
          tokensInputProcessBlock_[0],
          [this](edm::ProcessBlock const& processBlock, std::shared_ptr<long long int> const& previousCache) {
            auto returnValue = std::make_shared<long long int>(0);
            for (auto const& token : tokensInputProcessBlock_) {
              *returnValue += processBlock.get(token).value;
            }
            return returnValue;
          });
    }
  }

  TestFindProduct::~TestFindProduct() {}

  void TestFindProduct::analyze(edm::Event const& event, edm::EventSetup const&) {
    edm::Handle<IntProduct> h;
    edm::Handle<IntProduct> hToken;
    edm::Handle<edm::View<int>> hView;
    edm::Handle<edm::View<int>> hViewToken;

    std::vector<edm::EDGetTokenT<IntProduct>>::const_iterator iToken = tokens_.begin();
    for (std::vector<edm::InputTag>::const_iterator iter = inputTags_.begin(), iEnd = inputTags_.end(); iter != iEnd;
         ++iter, ++iToken) {
      if (getByTokenFirst_) {
        event.getByToken(*iToken, hToken);
        *hToken;
      }

      event.getByLabel(*iter, h);
      sum_ += h->value;

      event.getByToken(*iToken, hToken);
      if (h->value != hToken->value) {
        throw cms::Exception("TestFail")
            << "TestFindProduct::analyze getByLabel and getByToken return inconsistent results";
      }

      if (runProducerParameterCheck_) {
        edm::ParameterSet const* producerPset =
            edm::getProducerParameterSet(*hToken.provenance(), event.processHistory());
        int par = producerPset->getParameter<int>("ivalue");
        // These expected values are just from knowing the values in the
        // configuration files for this test.
        int expectedParameterValue = 3;
        if (!iter->process().empty()) {
          if (event.run() == 1) {
            expectedParameterValue = 1;
          } else {
            expectedParameterValue = 2;
          }
        }
        if (par != expectedParameterValue) {
          throw cms::Exception("TestFail") << "TestFindProduct::analyze unexpected value from producer parameter set";
        }
      }
    }
    iToken = tokensNotFound_.begin();
    for (std::vector<edm::InputTag>::const_iterator iter = inputTagsNotFound_.begin(), iEnd = inputTagsNotFound_.end();
         iter != iEnd;
         ++iter, ++iToken) {
      event.getByLabel(*iter, h);
      if (h.isValid()) {
        throw cms::Exception("TestFail")
            << "TestFindProduct::analyze: getByLabel found a product that should not be found "
            << h.provenance()->moduleLabel();
      }

      event.getByToken(*iToken, hToken);
      if (hToken.isValid()) {
        throw cms::Exception("TestFail")
            << "TestFindProduct::analyze: getByToken found a product that should not be found "
            << hToken.provenance()->moduleLabel();
      }
    }
    std::vector<edm::EDGetTokenT<edm::View<int>>>::const_iterator iTokenView = tokensView_.begin();
    for (std::vector<edm::InputTag>::const_iterator iter = inputTagsView_.begin(), iEnd = inputTagsView_.end();
         iter != iEnd;
         ++iter, ++iTokenView) {
      if (getByTokenFirst_) {
        event.getByToken(*iTokenView, hViewToken);
        *hViewToken;
      }

      event.getByLabel(*iter, hView);
      sum_ += hView->at(0);

      event.getByToken(*iTokenView, hViewToken);
      if (hView->at(0) != hViewToken->at(0)) {
        throw cms::Exception("TestFail")
            << "TestFindProduct::analyze getByLabel and getByToken return inconsistent results";
      }
    }

    // Get these also and add them into the sum
    edm::Handle<UInt64Product> h64;
    for (auto const& token : tokensUInt64_) {
      event.getByToken(token, h64);
      sum_ += h64->value;
    }

    if (expectedCache_ != 0) {
      std::tuple<edm::CacheHandle<int>, edm::CacheHandle<long long int>> valueTuple = processBlockCaches(event);
      {
        edm::CacheHandle<int> value = std::get<0>(valueTuple);
        if (*value != expectedCache_) {
          throw cms::Exception("TestFail") << "TestFindProduct::analyze 0 ProcessBlock cache has unexpected value "
                                           << *value << " expected = " << expectedCache_;
        }
      }
      {
        edm::CacheHandle<long long int> value = std::get<1>(valueTuple);
        if (*value != expectedCache_) {
          throw cms::Exception("TestFail") << "TestFindProduct::analyze 1 ProcessBlock cache has unexpected value "
                                           << *value << " expected = " << expectedCache_;
        }
      }
    }
  }

  void TestFindProduct::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

  void TestFindProduct::endLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const&) {
    // Get these also and add them into the sum
    edm::Handle<IntProduct> h;
    for (auto const& token : tokensEndLumi_) {
      lb.getByToken(token, h);
      sum_ += h->value;
    }
  }

  void TestFindProduct::beginRun(edm::Run const&, edm::EventSetup const&) {}

  void TestFindProduct::endRun(edm::Run const& run, edm::EventSetup const&) {
    // Get these also and add them into the sum
    edm::Handle<IntProduct> h;
    for (auto const& token : tokensEndRun_) {
      run.getByToken(token, h);
      sum_ += h->value;
    }
  }

  void TestFindProduct::beginProcessBlock(edm::ProcessBlock const& processBlock) {
    for (auto const& token : tokensBeginProcessBlock_) {
      sum_ += processBlock.get(token).value;
    }
    if (testGetterOfProducts_) {
      std::vector<edm::Handle<IntProduct>> handles;
      getterOfProducts_.fillHandles(processBlock, handles);
      for (auto const& intHandle : handles) {
        sum_ += intHandle->value;
      }
    }
  }

  void TestFindProduct::accessInputProcessBlock(edm::ProcessBlock const& processBlock) {
    for (auto const& token : tokensInputProcessBlock_) {
      int value = processBlock.get(token).value;
      sum_ += value;
    }
  }

  void TestFindProduct::endProcessBlock(edm::ProcessBlock const& processBlock) {
    std::vector<int> values;
    for (auto const& token : tokensEndProcessBlock_) {
      int value = processBlock.get(token).value;
      values.push_back(value);
      sum_ += value;
    }
    edm::Handle<IntProduct> h;
    unsigned int i = 0;
    for (auto val : values) {
      if (i < tokensEndProcessBlock2_.size()) {
        processBlock.getByToken(tokensEndProcessBlock2_[i], h);
        if (h->value != val + 2) {
          throw cms::Exception("TestFail") << "TestFindProduct::endProcessBlock 2, received unexpected value";
        }
      }
      if (i < tokensEndProcessBlock3_.size()) {
        processBlock.getByToken(tokensEndProcessBlock3_[i], h);
        if (h->value != val + 3) {
          throw cms::Exception("TestFail") << "TestFindProduct::endProcessBlock 3, received unexpected value";
        }
      }
      if (i < tokensEndProcessBlock4_.size()) {
        h = processBlock.getHandle(tokensEndProcessBlock4_[i]);
        if (h->value != val + 4) {
          throw cms::Exception("TestFail") << "TestFindProduct::endProcessBlock 4, received unexpected value";
        }
      }
      ++i;
    }
    if (testGetterOfProducts_) {
      std::vector<edm::Handle<IntProduct>> handles;
      getterOfProducts_.fillHandles(processBlock, handles);
      for (auto const& intHandle : handles) {
        sum_ += intHandle->value;
      }
    }
  }

  void TestFindProduct::endJob() {
    std::cout << "TestFindProduct sum = " << sum_ << std::endl;
    if (expectedSum_ != 0 && sum_ != expectedSum_) {
      throw cms::Exception("TestFail")
          << "TestFindProduct::endJob - Sum of test object values does not equal expected value";
    }
  }

}  // namespace edmtest

using edmtest::TestFindProduct;
DEFINE_FWK_MODULE(TestFindProduct);
