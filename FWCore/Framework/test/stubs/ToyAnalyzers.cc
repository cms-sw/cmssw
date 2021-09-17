
/*----------------------------------------------------------------------

Toy EDAnalyzers for testing purposes only.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
//
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"
//
#include <cassert>
#include <string>
#include <vector>

namespace edmtest {

  //--------------------------------------------------------------------
  //
  // Toy analyzers
  //
  //--------------------------------------------------------------------

  //--------------------------------------------------------------------
  //
  // every call to NonAnalyzer::analyze does nothing.
  //
  class NonAnalyzer : public edm::EDAnalyzer {
  public:
    explicit NonAnalyzer(edm::ParameterSet const& /*p*/) {}
    virtual ~NonAnalyzer() {}
    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  };

  void NonAnalyzer::analyze(edm::Event const&, edm::EventSetup const&) {}

  //--------------------------------------------------------------------
  //
  class IntTestAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    IntTestAnalyzer(edm::ParameterSet const& iPSet)
        : value_(iPSet.getUntrackedParameter<int>("valueMustMatch")),
          token_(consumes(iPSet.getUntrackedParameter<edm::InputTag>("moduleLabel"))) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<int>("valueMustMatch");
      desc.addUntracked<edm::InputTag>("moduleLabel");
      descriptions.addDefault(desc);
    }

    void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
      auto const& prod = iEvent.get(token_);
      if (prod.value != value_) {
        edm::ProductLabels labels;
        labelsForToken(token_, labels);
        throw cms::Exception("ValueMismatch")
            << "The value for \"" << labels.module << ":" << labels.productInstance << ":" << labels.process << "\" is "
            << prod.value << " but it was supposed to be " << value_;
      }
    }

  private:
    int const value_;
    edm::EDGetTokenT<IntProduct> const token_;
  };

  //--------------------------------------------------------------------
  //
  class MultipleIntsAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    MultipleIntsAnalyzer(edm::ParameterSet const& iPSet) {
      auto const& tags = iPSet.getUntrackedParameter<std::vector<edm::InputTag>>("getFromModules");
      for (auto const& tag : tags) {
        m_tokens.emplace_back(consumes(tag));
      }
    }

    void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const override {
      for (auto const& token : m_tokens) {
        (void)iEvent.getHandle(token);
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<std::vector<edm::InputTag>>("getFromModules");
      descriptions.addDefault(desc);
    }

  private:
    std::vector<edm::EDGetTokenT<IntProduct>> m_tokens;
  };

  //--------------------------------------------------------------------
  //
  template <typename T>
  class GenericAnalyzerT : public edm::global::EDAnalyzer<> {
  public:
    GenericAnalyzerT(edm::ParameterSet const& iPSet)
        : tokensBeginProcess_(edm::vector_transform(
              iPSet.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginProcess"),
              [this](edm::InputTag const& tag) { return this->consumes<T, edm::InProcess>(tag); })),
          tokensBeginRun_(
              edm::vector_transform(iPSet.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginRun"),
                                    [this](edm::InputTag const& tag) { return this->consumes<T, edm::InRun>(tag); })),
          tokensBeginLumi_(
              edm::vector_transform(iPSet.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginLumi"),
                                    [this](edm::InputTag const& tag) { return this->consumes<T, edm::InLumi>(tag); })),
          tokensEvent_(edm::vector_transform(iPSet.getUntrackedParameter<std::vector<edm::InputTag>>("srcEvent"),
                                             [this](edm::InputTag const& tag) { return this->consumes<T>(tag); })),
          tokensEndLumi_(
              edm::vector_transform(iPSet.getUntrackedParameter<std::vector<edm::InputTag>>("srcEndLumi"),
                                    [this](edm::InputTag const& tag) { return this->consumes<T, edm::InLumi>(tag); })),
          tokensEndRun_(
              edm::vector_transform(iPSet.getUntrackedParameter<std::vector<edm::InputTag>>("srcEndRun"),
                                    [this](edm::InputTag const& tag) { return this->consumes<T, edm::InRun>(tag); })),
          tokensEndProcess_(edm::vector_transform(
              iPSet.getUntrackedParameter<std::vector<edm::InputTag>>("srcEndProcess"),
              [this](edm::InputTag const& tag) { return this->consumes<T, edm::InProcess>(tag); })),
          shouldExist_(iPSet.getUntrackedParameter<bool>("inputShouldExist")),
          shouldBeMissing_(iPSet.getUntrackedParameter<bool>("inputShouldBeMissing")) {
      if (shouldExist_ and shouldBeMissing_) {
        throw cms::Exception("Configuration")
            << "inputShouldExist and inputShouldBeMissing both can not be set to True";
      }
    }

    void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const override {
      for (auto const& token : tokensEvent_) {
        if (shouldExist_) {
          (void)iEvent.get(token);
        } else if (shouldBeMissing_) {
          auto h = iEvent.getHandle(token);
          if (h.isValid()) {
            edm::EDConsumerBase::Labels labels;
            labelsForToken(token, labels);
            throw cms::Exception("ProductExists")
                << "Product " << labels.module << ":" << labels.productInstance << ":" << labels.process
                << " exists, but expectation was it should be missing";
          }
        }
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginProcess", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginRun", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginLumi", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcEvent", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcEndLumi", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcEndRun", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcEndProcess", std::vector<edm::InputTag>{});
      desc.addUntracked<bool>("inputShouldExist", false);
      desc.addUntracked<bool>("inputShouldBeMissing", false);
      descriptions.addDefault(desc);
    }

  private:
    const std::vector<edm::EDGetTokenT<T>> tokensBeginProcess_;
    const std::vector<edm::EDGetTokenT<T>> tokensBeginRun_;
    const std::vector<edm::EDGetTokenT<T>> tokensBeginLumi_;
    const std::vector<edm::EDGetTokenT<T>> tokensEvent_;
    const std::vector<edm::EDGetTokenT<T>> tokensEndLumi_;
    const std::vector<edm::EDGetTokenT<T>> tokensEndRun_;
    const std::vector<edm::EDGetTokenT<T>> tokensEndProcess_;
    const bool shouldExist_;
    const bool shouldBeMissing_;
  };
  using GenericIntsAnalyzer = GenericAnalyzerT<IntProduct>;
  using GenericUInt64Analyzer = GenericAnalyzerT<UInt64Product>;

  //--------------------------------------------------------------------
  //
  class IntConsumingAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    IntConsumingAnalyzer(edm::ParameterSet const& iPSet) {
      consumes<IntProduct>(iPSet.getUntrackedParameter<edm::InputTag>("getFromModule"));
    }

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<edm::InputTag>("getFromModule");
      descriptions.addDefault(desc);
    }
  };
  //--------------------------------------------------------------------
  //
  class IntFromRunConsumingAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    IntFromRunConsumingAnalyzer(edm::ParameterSet const& iPSet) {
      consumes<IntProduct, edm::InRun>(iPSet.getUntrackedParameter<edm::InputTag>("getFromModule"));
    }

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<edm::InputTag>("getFromModule");
      descriptions.addDefault(desc);
    }
  };
  //--------------------------------------------------------------------
  //
  class ConsumingStreamAnalyzer : public edm::stream::EDAnalyzer<> {
  public:
    ConsumingStreamAnalyzer(edm::ParameterSet const& iPSet)
        : value_(iPSet.getUntrackedParameter<int>("valueMustMatch")),
          moduleLabel_(iPSet.getUntrackedParameter<std::string>("moduleLabel"), "") {
      mayConsume<IntProduct>(moduleLabel_);
    }

    void analyze(edm::Event const& iEvent, edm::EventSetup const&) {
      edm::Handle<IntProduct> handle;
      iEvent.getByLabel(moduleLabel_, handle);
      if (handle->value != value_) {
        throw cms::Exception("ValueMissMatch") << "The value for \"" << moduleLabel_ << "\" is " << handle->value
                                               << " but it was supposed to be " << value_;
      }
    }

  private:
    int value_;
    edm::InputTag moduleLabel_;
  };

  //--------------------------------------------------------------------
  //
  class ConsumingOneSharedResourceAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
  public:
    ConsumingOneSharedResourceAnalyzer(edm::ParameterSet const& iPSet)
        : value_(iPSet.getUntrackedParameter<int>("valueMustMatch")),
          moduleLabel_(iPSet.getUntrackedParameter<edm::InputTag>("moduleLabel")) {
      mayConsume<IntProduct>(moduleLabel_);
      usesResource(iPSet.getUntrackedParameter<std::string>("resourceName"));
    }

    void analyze(edm::Event const& iEvent, edm::EventSetup const&) {
      edm::Handle<IntProduct> handle;
      iEvent.getByLabel(moduleLabel_, handle);
      if (handle->value != value_) {
        throw cms::Exception("ValueMissMatch") << "The value for \"" << moduleLabel_ << "\" is " << handle->value
                                               << " but it was supposed to be " << value_;
      }
    }

  private:
    int value_;
    edm::InputTag moduleLabel_;
  };

  //--------------------------------------------------------------------
  //
  class SCSimpleAnalyzer : public edm::EDAnalyzer {
  public:
    SCSimpleAnalyzer(edm::ParameterSet const&) {}

    virtual void analyze(edm::Event const& e, edm::EventSetup const&);
  };

  void SCSimpleAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&) {
    // Get the product back out; it should be sorted.
    edm::Handle<SCSimpleProduct> h;
    e.getByLabel("scs", h);
    assert(h.isValid());

    // Check the sorting. DO NOT DO THIS IN NORMAL CODE; we are
    // copying all the values out of the SortedCollection so we can
    // manipulate them via an interface different from
    // SortedCollection, just so that we can make sure the collection
    // is sorted.
    std::vector<Simple> after(h->begin(), h->end());
    typedef std::vector<Simple>::size_type size_type;

    // Verify that the vector *is* sorted.

    for (size_type i = 1, end = after.size(); i < end; ++i) {
      assert(after[i - 1].id() < after[i].id());
    }
  }

  //--------------------------------------------------------------------
  //
  // Consumes View<Simple>
  //
  class SimpleViewAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit SimpleViewAnalyzer(edm::ParameterSet const& p)
        : token_(consumes(p.getUntrackedParameter<edm::InputTag>("label"))),
          size_(p.getUntrackedParameter<unsigned int>("sizeMustMatch")),
          checkSize_(p.getUntrackedParameter<bool>("checkSize")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<bool>("checkSize", true);
      desc.addUntracked<unsigned int>("sizeMustMatch");
      desc.addUntracked<edm::InputTag>("label");
      descriptions.addDefault(desc);
    }

    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const&) const {
      if (checkSize_) {
        auto const& prod = e.get(token_);
        if (prod.size() != size_) {
          throw cms::Exception("SizeMismatch")
              << "Product size " << prod.size() << " differs from expectation " << size_;
        }
      }
    }

  private:
    edm::EDGetTokenT<edm::View<Simple>> const token_;
    unsigned int const size_;
    bool const checkSize_;
  };

  //--------------------------------------------------------------------
  //
  class DSVAnalyzer : public edm::EDAnalyzer {
  public:
    DSVAnalyzer(edm::ParameterSet const&) {}

    virtual void analyze(edm::Event const& e, edm::EventSetup const&);

  private:
    void do_sorted_stuff(edm::Event const& e);
    void do_unsorted_stuff(edm::Event const& e);
  };

  void DSVAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&) {
    do_sorted_stuff(e);
    do_unsorted_stuff(e);
  }

  void DSVAnalyzer::do_sorted_stuff(edm::Event const& e) {
    typedef DSVSimpleProduct product_type;
    typedef product_type::value_type detset;
    typedef detset::value_type value_type;
    // Get the product back out; it should be sorted.
    edm::Handle<product_type> h;
    e.getByLabel("dsv1", h);
    assert(h.isValid());

    // Check the sorting. DO NOT DO THIS IN NORMAL CODE; we are
    // copying all the values out of the DetSetVector's first DetSet so we can
    // manipulate them via an interface different from
    // DetSet, just so that we can make sure the collection
    // is sorted.
    std::vector<value_type> const& after = (h->end() - 1)->data;
    typedef std::vector<value_type>::size_type size_type;

    // Verify that the vector *is* sorted.

    for (size_type i = 1, end = after.size(); i < end; ++i) {
      assert(after[i - 1].data < after[i].data);
    }
  }

  void DSVAnalyzer::do_unsorted_stuff(edm::Event const& e) {
    typedef DSVWeirdProduct product_type;
    typedef product_type::value_type detset;
    typedef detset::value_type value_type;
    // Get the product back out; it should be unsorted.
    edm::Handle<product_type> h;
    e.getByLabel("dsv1", h);
    assert(h.isValid());

    // Check the sorting. DO NOT DO THIS IN NORMAL CODE; we are
    // copying all the values out of the DetSetVector's first DetSet so we can
    // manipulate them via an interface different from
    // DetSet, just so that we can make sure the collection
    // is not sorted.
    std::vector<value_type> const& after = (h->end() - 1)->data;
    typedef std::vector<value_type>::size_type size_type;

    // Verify that the vector is reverse-sorted.

    for (size_type i = 1, end = after.size(); i < end; ++i) {
      assert(after[i - 1].data > after[i].data);
    }
  }
}  // namespace edmtest

using edmtest::ConsumingOneSharedResourceAnalyzer;
using edmtest::ConsumingStreamAnalyzer;
using edmtest::DSVAnalyzer;
using edmtest::IntConsumingAnalyzer;
using edmtest::IntTestAnalyzer;
using edmtest::MultipleIntsAnalyzer;
using edmtest::NonAnalyzer;
using edmtest::SCSimpleAnalyzer;
using edmtest::SimpleViewAnalyzer;
DEFINE_FWK_MODULE(NonAnalyzer);
DEFINE_FWK_MODULE(IntTestAnalyzer);
DEFINE_FWK_MODULE(MultipleIntsAnalyzer);
DEFINE_FWK_MODULE(edmtest::GenericIntsAnalyzer);
DEFINE_FWK_MODULE(edmtest::GenericUInt64Analyzer);
DEFINE_FWK_MODULE(IntConsumingAnalyzer);
DEFINE_FWK_MODULE(edmtest::IntFromRunConsumingAnalyzer);
DEFINE_FWK_MODULE(ConsumingStreamAnalyzer);
DEFINE_FWK_MODULE(ConsumingOneSharedResourceAnalyzer);
DEFINE_FWK_MODULE(SCSimpleAnalyzer);
DEFINE_FWK_MODULE(SimpleViewAnalyzer);
DEFINE_FWK_MODULE(DSVAnalyzer);
