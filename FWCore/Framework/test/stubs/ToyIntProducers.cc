
/*----------------------------------------------------------------------

Toy EDProducers of Ints for testing purposes only.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
//
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/limited/EDProducer.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/TypeMatch.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"
//
#include <cassert>
#include <string>
#include <vector>
#include <unistd.h>

namespace edmtest {

  //--------------------------------------------------------------------
  //
  // Int producers
  //
  //--------------------------------------------------------------------

  //--------------------------------------------------------------------
  //
  // throws an exception.
  // Announces an IntProduct but does not produce one;
  // every call to FailingProducer::produce throws a cms exception
  //
  class FailingProducer : public edm::global::EDProducer<> {
  public:
    explicit FailingProducer(edm::ParameterSet const& /*p*/) { produces<IntProduct>(); }
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;
  };

  void FailingProducer::produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const {
    // We throw an edm exception with a configurable action.
    throw edm::Exception(edm::errors::NotFound) << "Intentional 'NotFound' exception for testing purposes\n";
  }
  //--------------------------------------------------------------------
  //
  // throws an exception.
  // Announces an IntProduct but does not produce one;
  // every call to FailingInLumiProducer::produce throws a cms exception
  //
  class FailingInLumiProducer : public edm::global::EDProducer<edm::BeginLuminosityBlockProducer> {
  public:
    explicit FailingInLumiProducer(edm::ParameterSet const& /*p*/) {
      produces<IntProduct, edm::Transition::BeginLuminosityBlock>();
    }
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

    void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override;
  };

  void FailingInLumiProducer::produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const {}
  void FailingInLumiProducer::globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const {
    // We throw an edm exception with a configurable action.
    throw edm::Exception(edm::errors::NotFound) << "Intentional 'NotFound' exception for testing purposes\n";
  }

  //--------------------------------------------------------------------
  //
  // throws an exception.
  // Announces an IntProduct but does not produce one;
  // every call to FailingProducer::produce throws a cms exception
  //
  class FailingInRunProducer : public edm::global::EDProducer<edm::BeginRunProducer> {
  public:
    explicit FailingInRunProducer(edm::ParameterSet const& /*p*/) { produces<IntProduct, edm::Transition::BeginRun>(); }
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

    void globalBeginRunProduce(edm::Run&, edm::EventSetup const&) const override;
  };

  void FailingInRunProducer::produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const {}
  void FailingInRunProducer::globalBeginRunProduce(edm::Run&, edm::EventSetup const&) const {
    // We throw an edm exception with a configurable action.
    throw edm::Exception(edm::errors::NotFound) << "Intentional 'NotFound' exception for testing purposes\n";
  }

  //--------------------------------------------------------------------
  //
  // Announces an IntProduct but does not produce one;
  // every call to NonProducer::produce does nothing.
  //
  class NonProducer : public edm::global::EDProducer<> {
  public:
    explicit NonProducer(edm::ParameterSet const& /*p*/) { produces<IntProduct>(); }
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;
  };

  void NonProducer::produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const {}

  //--------------------------------------------------------------------
  //
  // Produces an IntProduct instance.
  //
  // NOTE: this really should be a global::EDProducer<> but for testing we use stream
  class IntProducer : public edm::stream::EDProducer<> {
  public:
    explicit IntProducer(edm::ParameterSet const& p)
        : token_{produces<IntProduct>()}, value_(p.getParameter<int>("ivalue")) {}
    void produce(edm::Event& e, edm::EventSetup const& c) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int>("ivalue");
      descriptions.addDefault(desc);
    }

  private:
    edm::EDPutTokenT<IntProduct> token_;
    int value_;
  };

  void IntProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    e.emplace(token_, value_);
  }

  //--------------------------------------------------------------------
  //
  // Produces an IntProduct instance.
  //
  class IntOneSharedProducer : public edm::one::EDProducer<edm::one::SharedResources> {
  public:
    explicit IntOneSharedProducer(edm::ParameterSet const& p) : value_(p.getParameter<int>("ivalue")) {
      produces<IntProduct>();
      for (auto const& r : p.getUntrackedParameter<std::vector<std::string>>("resourceNames")) {
        usesResource(r);
      }
    }
    explicit IntOneSharedProducer(int i) : value_(i) {
      produces<IntProduct>();
      usesResource("IntOneShared");
    }
    void produce(edm::Event& e, edm::EventSetup const& c) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& iConfig) {
      edm::ParameterSetDescription desc;
      desc.add<int>("ivalue");
      desc.addUntracked<std::vector<std::string>>("resourceNames", std::vector<std::string>{});

      iConfig.addDefault(desc);
    }

  private:
    int value_;
  };

  void IntOneSharedProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    e.put(std::make_unique<IntProduct>(value_));
  }

  //--------------------------------------------------------------------
  //
  // Produces an IntProduct instance.
  //
  //
  class BusyWaitIntProducer : public edm::global::EDProducer<> {
  public:
    explicit BusyWaitIntProducer(edm::ParameterSet const& p)
        : token_{produces<IntProduct>()},
          value_(p.getParameter<int>("ivalue")),
          iterations_(p.getParameter<unsigned int>("iterations")),
          pi_(std::acos(-1)),
          lumiNumberToThrow_(p.getParameter<unsigned int>("lumiNumberToThrow")) {}

    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    const edm::EDPutTokenT<IntProduct> token_;
    const int value_;
    const unsigned int iterations_;
    const double pi_;
    const unsigned int lumiNumberToThrow_;
  };

  void BusyWaitIntProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    double sum = 0.;
    const double stepSize = pi_ / iterations_;
    for (unsigned int i = 0; i < iterations_; ++i) {
      sum += stepSize * cos(i * stepSize);
    }

    e.emplace(token_, value_ + sum);

    if (e.luminosityBlock() == lumiNumberToThrow_) {
      throw cms::Exception("Test");
    }
  }

  void BusyWaitIntProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<int>("ivalue");
    desc.add<unsigned int>("iterations");
    desc.add<unsigned int>("lumiNumberToThrow", 0);
    descriptions.addDefault(desc);
  }

  //--------------------------------------------------------------------
  class BusyWaitIntLimitedProducer : public edm::limited::EDProducer<> {
  public:
    explicit BusyWaitIntLimitedProducer(edm::ParameterSet const& p)
        : edm::limited::EDProducerBase(p),
          edm::limited::EDProducer<>(p),
          token_{produces<IntProduct>()},
          value_(p.getParameter<int>("ivalue")),
          iterations_(p.getParameter<unsigned int>("iterations")),
          pi_(std::acos(-1)) {}

    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

  private:
    const edm::EDPutTokenT<IntProduct> token_;
    const int value_;
    const unsigned int iterations_;
    const double pi_;
    mutable std::atomic<unsigned int> reentrancy_{0};
  };

  void BusyWaitIntLimitedProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    auto v = ++reentrancy_;
    if (v > concurrencyLimit()) {
      --reentrancy_;
      throw cms::Exception("NotLimited", "produce called to many times concurrently.");
    }

    double sum = 0.;
    const double stepSize = pi_ / iterations_;
    for (unsigned int i = 0; i < iterations_; ++i) {
      sum += stepSize * cos(i * stepSize);
    }

    e.emplace(token_, value_ + sum);
    --reentrancy_;
  }

  //--------------------------------------------------------------------
  class BusyWaitIntOneSharedProducer : public edm::one::EDProducer<edm::one::SharedResources> {
  public:
    explicit BusyWaitIntOneSharedProducer(edm::ParameterSet const& p)
        : value_(p.getParameter<int>("ivalue")),
          iterations_(p.getParameter<unsigned int>("iterations")),
          pi_(std::acos(-1)) {
      for (auto const& r : p.getUntrackedParameter<std::vector<std::string>>("resourceNames")) {
        usesResource(r);
      }
      produces<IntProduct>();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& iConfig) {
      edm::ParameterSetDescription desc;
      desc.add<int>("ivalue");
      desc.add<unsigned int>("iterations");
      desc.addUntracked<std::vector<std::string>>("resourceNames", std::vector<std::string>{});

      iConfig.addDefault(desc);
    }

    void produce(edm::Event& e, edm::EventSetup const& c) override;

  private:
    const int value_;
    const unsigned int iterations_;
    const double pi_;
  };

  void BusyWaitIntOneSharedProducer::produce(edm::Event& e, edm::EventSetup const&) {
    double sum = 0.;
    const double stepSize = pi_ / iterations_;
    for (unsigned int i = 0; i < iterations_; ++i) {
      sum += stepSize * cos(i * stepSize);
    }

    e.put(std::make_unique<IntProduct>(value_ + sum));
  }

  //--------------------------------------------------------------------

  class ConsumingIntProducer : public edm::stream::EDProducer<> {
  public:
    explicit ConsumingIntProducer(edm::ParameterSet const& p)
        : getterOfTriggerResults_(edm::TypeMatch(), this),
          token_{produces<IntProduct>()},
          value_(p.getParameter<int>("ivalue")) {
      // not used, only exists to test PathAndConsumesOfModules
      consumes<edm::TriggerResults>(edm::InputTag("TriggerResults"));
      callWhenNewProductsRegistered(getterOfTriggerResults_);
    }
    void produce(edm::Event& e, edm::EventSetup const& c) override;

  private:
    edm::GetterOfProducts<edm::TriggerResults> getterOfTriggerResults_;
    const edm::EDPutTokenT<IntProduct> token_;
    const int value_;
  };

  void ConsumingIntProducer::produce(edm::Event& e, edm::EventSetup const&) { e.emplace(token_, value_); }

  //--------------------------------------------------------------------
  //
  // Produces an IntProduct instance whose value is the event number,
  // rather than the value of a configured parameter.
  //
  class EventNumberIntProducer : public edm::global::EDProducer<> {
  public:
    explicit EventNumberIntProducer(edm::ParameterSet const&) : token_{produces<UInt64Product>()} {}
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

  private:
    const edm::EDPutTokenT<UInt64Product> token_;
  };

  void EventNumberIntProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    // EventSetup is not used.
    e.emplace(token_, e.id().event());
  }

  //--------------------------------------------------------------------
  //
  // Produces a TransientIntProduct instance.
  //
  class TransientIntProducer : public edm::global::EDProducer<> {
  public:
    explicit TransientIntProducer(edm::ParameterSet const& p)
        : token_{produces<TransientIntProduct>()}, value_(p.getParameter<int>("ivalue")) {}
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

  private:
    const edm::EDPutTokenT<TransientIntProduct> token_;
    const int value_;
  };

  void TransientIntProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    // EventSetup is not used.
    e.emplace(token_, value_);
  }

  //--------------------------------------------------------------------
  //
  // Produces a IntProduct instance from a TransientIntProduct
  //
  class IntProducerFromTransient : public edm::global::EDProducer<> {
  public:
    explicit IntProducerFromTransient(edm::ParameterSet const&)
        : putToken_{produces<IntProduct>()}, getToken_{consumes(edm::InputTag{"TransientThing"})} {}
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

  private:
    const edm::EDPutTokenT<IntProduct> putToken_;
    const edm::EDGetTokenT<TransientIntProduct> getToken_;
  };

  void IntProducerFromTransient::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    // EventSetup is not used.
    auto result = e.getHandle(getToken_);
    assert(result);
    e.emplace(putToken_, result.product()->value);
  }

  //--------------------------------------------------------------------
  //
  // Produces a TransientIntParent instance.
  //
  class TransientIntParentProducer : public edm::global::EDProducer<> {
  public:
    explicit TransientIntParentProducer(edm::ParameterSet const& p)
        : token_{produces<TransientIntParent>()}, value_(p.getParameter<int>("ivalue")) {}
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

  private:
    const edm::EDPutTokenT<TransientIntParent> token_;
    const int value_;
  };

  void TransientIntParentProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    // EventSetup is not used.
    e.emplace(token_, value_);
  }

  //--------------------------------------------------------------------
  //
  // Produces a IntProduct instance from a TransientIntParent
  //
  class IntProducerFromTransientParent : public edm::global::EDProducer<> {
  public:
    explicit IntProducerFromTransientParent(edm::ParameterSet const& p)
        : putToken_{produces<IntProduct>()}, getToken_{consumes(p.getParameter<edm::InputTag>("src"))} {}
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

  private:
    const edm::EDPutTokenT<IntProduct> putToken_;
    const edm::EDGetTokenT<TransientIntParent> getToken_;
  };

  void IntProducerFromTransientParent::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    // EventSetup is not used.
    e.emplace(putToken_, e.get(getToken_).value);
  }

  //--------------------------------------------------------------------
  //
  // Produces an Int16_tProduct instance.
  //
  class Int16_tProducer : public edm::global::EDProducer<> {
  public:
    explicit Int16_tProducer(edm::ParameterSet const& p)
        : token_{produces<Int16_tProduct>()}, value_(p.getParameter<int>("ivalue")) {}
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

  private:
    const edm::EDPutTokenT<Int16_tProduct> token_;
    const int16_t value_;
  };

  void Int16_tProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    // EventSetup is not used.
    e.emplace(token_, value_);
  }

  //
  // Produces an IntProduct instance, using an IntProduct as input.
  //

  class AddIntsProducer : public edm::global::EDProducer<> {
  public:
    explicit AddIntsProducer(edm::ParameterSet const& p)
        : putToken_{produces<IntProduct>()},
          otherPutToken_{produces<IntProduct>("other")},
          onlyGetOnEvent_(p.getUntrackedParameter<unsigned int>("onlyGetOnEvent")) {
      auto const& labels = p.getParameter<std::vector<edm::InputTag>>("labels");
      for (auto const& label : labels) {
        tokens_.emplace_back(consumes(label));
      }
    }
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<unsigned int>("onlyGetOnEvent", 0u);
      desc.add<std::vector<edm::InputTag>>("labels");
      descriptions.addDefault(desc);
    }

  private:
    std::vector<edm::EDGetTokenT<IntProduct>> tokens_;
    const edm::EDPutTokenT<IntProduct> putToken_;
    const edm::EDPutTokenT<IntProduct> otherPutToken_;
    unsigned int onlyGetOnEvent_;
  };

  void AddIntsProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    // EventSetup is not used.
    int value = 0;

    if (onlyGetOnEvent_ == 0u || e.eventAuxiliary().event() == onlyGetOnEvent_) {
      for (auto const& token : tokens_) {
        value += e.get(token).value;
      }
    }
    e.emplace(putToken_, value);
    e.emplace(otherPutToken_, value);
  }

  //
  // Produces an IntProduct instance, using many IntProducts as input.
  //

  class AddAllIntsProducer : public edm::global::EDProducer<> {
  public:
    explicit AddAllIntsProducer(edm::ParameterSet const& p) : putToken_{produces()} {
      getter_ = edm::GetterOfProducts<IntProduct>(edm::TypeMatch(), this);
      callWhenNewProductsRegistered(*getter_);
    }
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addDefault(desc);
    }

  private:
    const edm::EDPutTokenT<int> putToken_;
    std::optional<edm::GetterOfProducts<IntProduct>> getter_;
  };

  void AddAllIntsProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    std::vector<edm::Handle<IntProduct>> ints;
    getter_->fillHandles(e, ints);

    int value = 0;
    for (auto const& i : ints) {
      value += i->value;
    }

    e.emplace(putToken_, value);
  }

  //
  // Produces multiple IntProduct products
  //

  class ManyIntProducer : public edm::global::EDProducer<> {
  public:
    explicit ManyIntProducer(edm::ParameterSet const& p)
        : tokenValues_{vector_transform(
              p.getParameter<std::vector<edm::ParameterSet>>("values"),
              [this](edm::ParameterSet const& pset) {
                auto const& branchAlias = pset.getParameter<std::string>("branchAlias");
                if (not branchAlias.empty()) {
                  return TokenValue{
                      produces<IntProduct>(pset.getParameter<std::string>("instance")).setBranchAlias(branchAlias),
                      pset.getParameter<int>("value")};
                }
                return TokenValue{produces<IntProduct>(pset.getParameter<std::string>("instance")),
                                  pset.getParameter<int>("value")};
              })},
          transientTokenValues_{vector_transform(
              p.getParameter<std::vector<edm::ParameterSet>>("transientValues"),
              [this](edm::ParameterSet const& pset) {
                auto const& branchAlias = pset.getParameter<std::string>("branchAlias");
                if (not branchAlias.empty()) {
                  return TransientTokenValue{produces<TransientIntProduct>(pset.getParameter<std::string>("instance"))
                                                 .setBranchAlias(branchAlias),
                                             pset.getParameter<int>("value")};
                }
                return TransientTokenValue{produces<TransientIntProduct>(pset.getParameter<std::string>("instance")),
                                           pset.getParameter<int>("value")};
              })},
          throw_{p.getUntrackedParameter<bool>("throw")} {
      tokenValues_.push_back(TokenValue{produces<IntProduct>(), p.getParameter<int>("ivalue")});
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int>("ivalue");
      desc.addUntracked<bool>("throw", false);

      {
        edm::ParameterSetDescription pset;
        pset.add<std::string>("instance");
        pset.add<int>("value");
        pset.add<std::string>("branchAlias", "");
        desc.addVPSet("values", pset, std::vector<edm::ParameterSet>{});
        desc.addVPSet("transientValues", pset, std::vector<edm::ParameterSet>{});
      }

      descriptions.addDefault(desc);
    }

    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

  private:
    struct TokenValue {
      edm::EDPutTokenT<IntProduct> token;
      int value;
    };
    std::vector<TokenValue> tokenValues_;

    struct TransientTokenValue {
      edm::EDPutTokenT<TransientIntProduct> token;
      int value;
    };
    std::vector<TransientTokenValue> transientTokenValues_;

    bool throw_;
  };

  void ManyIntProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    if (throw_) {
      throw edm::Exception(edm::errors::NotFound) << "Intentional 'NotFound' exception for testing purposes\n";
    }

    // EventSetup is not used.
    for (auto const& tv : tokenValues_) {
      e.emplace(tv.token, tv.value);
    }
    for (auto const& tv : transientTokenValues_) {
      e.emplace(tv.token, tv.value);
    }
  }

  //
  // Produces multiple IntProduct products based on other products
  //

  class ManyIntWhenRegisteredProducer : public edm::global::EDProducer<> {
  public:
    explicit ManyIntWhenRegisteredProducer(edm::ParameterSet const& p)
        : sourceLabel_(p.getParameter<std::string>("src")) {
      callWhenNewProductsRegistered([=](edm::BranchDescription const& iBranch) {
        if (iBranch.moduleLabel() == sourceLabel_) {
          if (iBranch.branchType() != edm::InEvent) {
            throw edm::Exception(edm::errors::UnimplementedFeature)
                << "ManyIntWhenRegisteredProducer supports only event branches";
          }

          this->tokens_.push_back(
              Tokens{this->consumes<IntProduct>(
                         edm::InputTag{iBranch.moduleLabel(), iBranch.productInstanceName(), iBranch.processName()}),
                     this->produces<IntProduct>(iBranch.productInstanceName())});
        }
      });
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("src");
      descriptions.addDefault(desc);
    }

    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

  private:
    struct Tokens {
      edm::EDGetTokenT<IntProduct> get;
      edm::EDPutTokenT<IntProduct> put;
    };

    std::string sourceLabel_;
    std::vector<Tokens> tokens_;
  };

  void ManyIntWhenRegisteredProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    for (auto const& toks : tokens_) {
      e.emplace(toks.put, e.get(toks.get));
    }
  };

  //--------------------------------------------------------------------

  class NonEventIntProducer : public edm::global::EDProducer<edm::Accumulator,
                                                             edm::BeginRunProducer,
                                                             edm::BeginLuminosityBlockProducer,
                                                             edm::EndLuminosityBlockProducer,
                                                             edm::EndRunProducer,
                                                             edm::BeginProcessBlockProducer,
                                                             edm::EndProcessBlockProducer,
                                                             edm::InputProcessBlockCache<>> {
  public:
    explicit NonEventIntProducer(edm::ParameterSet const& p)
        : bpbToken_{produces<IntProduct, edm::Transition::BeginProcessBlock>("beginProcessBlock")},
          brToken_{produces<IntProduct, edm::Transition::BeginRun>("beginRun")},
          blToken_{produces<IntProduct, edm::Transition::BeginLuminosityBlock>("beginLumi")},
          elToken_{produces<IntProduct, edm::Transition::EndLuminosityBlock>("endLumi")},
          erToken_{produces<IntProduct, edm::Transition::EndRun>("endRun")},
          epbToken_{produces<IntProduct, edm::Transition::EndProcessBlock>("endProcessBlock")},
          value_(p.getParameter<int>("ivalue")),
          sleepTime_(p.getParameter<unsigned int>("sleepTime")),
          bpbExpect_{p.getUntrackedParameter<int>("expectBeginProcessBlock")},
          brExpect_{p.getUntrackedParameter<int>("expectBeginRun")},
          blExpect_{p.getUntrackedParameter<int>("expectBeginLuminosityBlock")},
          elExpect_{p.getUntrackedParameter<int>("expectEndLuminosityBlock")},
          erExpect_{p.getUntrackedParameter<int>("expectEndRun")},
          epbExpect_{p.getUntrackedParameter<int>("expectEndProcessBlock")},
          aipbExpect_{p.getUntrackedParameter<int>("expectAccessInputProcessBlock")} {
      {
        auto tag = p.getParameter<edm::InputTag>("consumesBeginProcessBlock");
        if (not tag.label().empty()) {
          bpbGet_ = consumes<edm::InProcess>(tag);
        }
      }
      {
        auto tag = p.getParameter<edm::InputTag>("consumesBeginRun");
        if (not tag.label().empty()) {
          brGet_ = consumes<edm::InRun>(tag);
        }
      }
      {
        auto tag = p.getParameter<edm::InputTag>("consumesBeginLuminosityBlock");
        if (not tag.label().empty()) {
          blGet_ = consumes<edm::InLumi>(tag);
        }
      }
      {
        auto tag = p.getParameter<edm::InputTag>("consumesEndLuminosityBlock");
        if (not tag.label().empty()) {
          elGet_ = consumes<edm::InLumi>(tag);
        }
      }
      {
        auto tag = p.getParameter<edm::InputTag>("consumesEndRun");
        if (not tag.label().empty()) {
          erGet_ = consumes<edm::InRun>(tag);
        }
      }
      {
        auto tag = p.getParameter<edm::InputTag>("consumesEndProcessBlock");
        if (not tag.label().empty()) {
          epbGet_ = consumes<edm::InProcess>(tag);
        }
      }
      {
        auto tag = p.getParameter<edm::InputTag>("consumesAccessInputProcessBlock");
        if (not tag.label().empty()) {
          aipbGet_ = consumes<edm::InProcess>(tag);
        }
      }
    }
    void accumulate(edm::StreamID iID, edm::Event const& e, edm::EventSetup const& c) const override;
    void beginProcessBlockProduce(edm::ProcessBlock&) override;
    void endProcessBlockProduce(edm::ProcessBlock&) override;
    void globalBeginRunProduce(edm::Run& e, edm::EventSetup const&) const override;
    void globalEndRunProduce(edm::Run& e, edm::EventSetup const&) const override;
    void globalBeginLuminosityBlockProduce(edm::LuminosityBlock& e, edm::EventSetup const&) const override;
    void globalEndLuminosityBlockProduce(edm::LuminosityBlock& e, edm::EventSetup const&) const override;
    void accessInputProcessBlock(edm::ProcessBlock const&) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& conf) {
      edm::ParameterSetDescription desc;
      desc.add<int>("ivalue", 0);
      desc.add<unsigned int>("sleepTime", 0);
      desc.add<edm::InputTag>("consumesBeginProcessBlock", {});
      desc.addUntracked<int>("expectBeginProcessBlock", 0);
      desc.add<edm::InputTag>("consumesEndProcessBlock", {});
      desc.addUntracked<int>("expectEndProcessBlock", 0);
      desc.add<edm::InputTag>("consumesAccessInputProcessBlock", {});
      desc.addUntracked<int>("expectAccessInputProcessBlock", 0);
      desc.add<edm::InputTag>("consumesBeginRun", {});
      desc.addUntracked<int>("expectBeginRun", 0);
      desc.add<edm::InputTag>("consumesEndRun", {});
      desc.addUntracked<int>("expectEndRun", 0);
      desc.add<edm::InputTag>("consumesBeginLuminosityBlock", {});
      desc.addUntracked<int>("expectBeginLuminosityBlock", 0);
      desc.add<edm::InputTag>("consumesEndLuminosityBlock", {});
      desc.addUntracked<int>("expectEndLuminosityBlock", 0);

      conf.addDefault(desc);
    }

  private:
    void check(IntProduct, int) const;
    const edm::EDPutTokenT<IntProduct> bpbToken_;
    const edm::EDPutTokenT<IntProduct> brToken_;
    const edm::EDPutTokenT<IntProduct> blToken_;
    const edm::EDPutTokenT<IntProduct> elToken_;
    const edm::EDPutTokenT<IntProduct> erToken_;
    const edm::EDPutTokenT<IntProduct> epbToken_;
    const int value_;
    const unsigned int sleepTime_;
    edm::EDGetTokenT<IntProduct> bpbGet_;
    edm::EDGetTokenT<IntProduct> brGet_;
    edm::EDGetTokenT<IntProduct> blGet_;
    edm::EDGetTokenT<IntProduct> elGet_;
    edm::EDGetTokenT<IntProduct> erGet_;
    edm::EDGetTokenT<IntProduct> epbGet_;
    edm::EDGetTokenT<IntProduct> aipbGet_;
    const int bpbExpect_;
    const int brExpect_;
    const int blExpect_;
    const int elExpect_;
    const int erExpect_;
    const int epbExpect_;
    const int aipbExpect_;
  };

  void NonEventIntProducer::accumulate(edm::StreamID iID, edm::Event const& e, edm::EventSetup const&) const {}
  void NonEventIntProducer::beginProcessBlockProduce(edm::ProcessBlock& processBlock) {
    if (not bpbGet_.isUninitialized()) {
      check(processBlock.get(bpbGet_), bpbExpect_);
    }
    if (sleepTime_ > 0) {
      // These sleeps are here to force modules to run concurrently
      // in multi-threaded processes. Otherwise, the modules are so
      // fast it is hard to tell whether a module finishes before the
      // the Framework starts the next module.
      usleep(sleepTime_);
    }
    processBlock.emplace(bpbToken_, value_);
  }
  void NonEventIntProducer::endProcessBlockProduce(edm::ProcessBlock& processBlock) {
    if (not epbGet_.isUninitialized()) {
      check(processBlock.get(epbGet_), epbExpect_);
    }
    if (sleepTime_ > 0) {
      usleep(sleepTime_);
    }
    processBlock.emplace(epbToken_, value_);
  }
  void NonEventIntProducer::accessInputProcessBlock(edm::ProcessBlock const& processBlock) {
    if (not aipbGet_.isUninitialized()) {
      check(processBlock.get(aipbGet_), aipbExpect_);
    }
    if (sleepTime_ > 0) {
      usleep(sleepTime_);
    }
  }
  void NonEventIntProducer::globalBeginRunProduce(edm::Run& r, edm::EventSetup const&) const {
    if (not brGet_.isUninitialized()) {
      check(r.get(brGet_), brExpect_);
    }
    if (sleepTime_ > 0) {
      usleep(sleepTime_);
    }
    r.emplace(brToken_, value_);
  }
  void NonEventIntProducer::globalEndRunProduce(edm::Run& r, edm::EventSetup const&) const {
    if (not erGet_.isUninitialized()) {
      check(r.get(erGet_), erExpect_);
    }
    if (sleepTime_ > 0) {
      usleep(sleepTime_);
    }
    r.emplace(erToken_, value_);
  }
  void NonEventIntProducer::globalBeginLuminosityBlockProduce(edm::LuminosityBlock& l, edm::EventSetup const&) const {
    if (not blGet_.isUninitialized()) {
      check(l.get(blGet_), blExpect_);
    }
    if (sleepTime_ > 0) {
      usleep(sleepTime_);
    }
    l.emplace(blToken_, value_);
  }
  void NonEventIntProducer::globalEndLuminosityBlockProduce(edm::LuminosityBlock& l, edm::EventSetup const&) const {
    if (not elGet_.isUninitialized()) {
      check(l.get(elGet_), elExpect_);
    }
    if (sleepTime_ > 0) {
      usleep(sleepTime_);
    }
    l.emplace(elToken_, value_);
  }
  void NonEventIntProducer::check(IntProduct iProd, int iExpect) const {
    if (iExpect != iProd.value) {
      throw cms::Exception("BadValue") << "expected " << iExpect << " but got " << iProd.value;
    }
  }

  //--------------------------------------------------------------------
  //
  // Produces an IntProduct in ProcessBlock at beginProcessBlock
  //
  class IntProducerBeginProcessBlock : public edm::global::EDProducer<edm::BeginProcessBlockProducer> {
  public:
    explicit IntProducerBeginProcessBlock(edm::ParameterSet const& p)
        : token_{produces<IntProduct, edm::Transition::BeginProcessBlock>()}, value_(p.getParameter<int>("ivalue")) {}
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {}
    void beginProcessBlockProduce(edm::ProcessBlock&) override;

  private:
    edm::EDPutTokenT<IntProduct> token_;
    int value_;
  };

  void IntProducerBeginProcessBlock::beginProcessBlockProduce(edm::ProcessBlock& processBlock) {
    processBlock.emplace(token_, value_);
  }

  //--------------------------------------------------------------------
  //
  // Produces an IntProduct in ProcessBlock at endProcessBlock
  //
  class IntProducerEndProcessBlock : public edm::global::EDProducer<edm::EndProcessBlockProducer> {
  public:
    explicit IntProducerEndProcessBlock(edm::ParameterSet const& p)
        : token_{produces<IntProduct, edm::Transition::EndProcessBlock>()},
          token2_{produces<IntProduct, edm::Transition::EndProcessBlock>("two")},
          token3_{produces<IntProduct, edm::Transition::EndProcessBlock>("three")},
          token4_{produces<IntProduct, edm::Transition::EndProcessBlock>("four")},
          value_(p.getParameter<int>("ivalue")) {}
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {}
    void endProcessBlockProduce(edm::ProcessBlock&) override;

  private:
    edm::EDPutTokenT<IntProduct> token_;
    edm::EDPutToken token2_;
    edm::EDPutTokenT<IntProduct> token3_;
    edm::EDPutToken token4_;
    int value_;
  };

  void IntProducerEndProcessBlock::endProcessBlockProduce(edm::ProcessBlock& processBlock) {
    processBlock.emplace(token_, value_);
    processBlock.emplace<IntProduct>(token2_, value_ + 2);
    processBlock.put(token3_, std::make_unique<IntProduct>(value_ + 3));
    processBlock.put(token4_, std::make_unique<IntProduct>(value_ + 4));
  }

  //--------------------------------------------------------------------
  //
  // Produces an TransientIntProduct in ProcessBlock at endProcessBlock
  //
  class TransientIntProducerEndProcessBlock : public edm::global::EDProducer<edm::EndProcessBlockProducer> {
  public:
    explicit TransientIntProducerEndProcessBlock(edm::ParameterSet const& p)
        : token_{produces<TransientIntProduct, edm::Transition::EndProcessBlock>()},
          value_(p.getParameter<int>("ivalue")) {}
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {}
    void endProcessBlockProduce(edm::ProcessBlock&) override;

  private:
    edm::EDPutTokenT<TransientIntProduct> token_;
    int value_;
  };

  void TransientIntProducerEndProcessBlock::endProcessBlockProduce(edm::ProcessBlock& processBlock) {
    processBlock.emplace(token_, value_);
  }

  //--------------------------------------------------------------------
  //
  // Produces an IntProduct instance, the module must get run, otherwise an exception is thrown
  class MustRunIntProducer : public edm::global::EDProducer<> {
  public:
    explicit MustRunIntProducer(edm::ParameterSet const& p)
        : moduleLabel_{p.getParameter<std::string>("@module_label")},
          token_{produces<IntProduct>()},
          value_(p.getParameter<int>("ivalue")),
          produce_{p.getParameter<bool>("produce")},
          mustRunEvent_{p.getParameter<bool>("mustRunEvent")} {}
    ~MustRunIntProducer() {
      if (not wasRunEndJob_) {
        throw cms::Exception("NotRun") << "This module (" << moduleLabel_
                                       << ") should have run for endJob transition, but it did not";
      }
    }
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int>("ivalue");
      desc.add<bool>("produce", true);
      desc.add<bool>("mustRunEvent", true)
          ->setComment(
              "If set to false, the endJob() is still required to be called to check that the module was not deleted "
              "early on");
      descriptions.addDefault(desc);
    }
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override {
      wasRunEvent_ = true;
      if (produce_) {
        e.emplace(token_, value_);
      }
    }
    void endJob() override {
      wasRunEndJob_ = true;
      if (mustRunEvent_ and not wasRunEvent_) {
        throw cms::Exception("NotRun") << "This module should have run for event transitions, but it did not";
      }
    }

  private:
    const std::string moduleLabel_;
    const edm::EDPutTokenT<IntProduct> token_;
    const int value_;
    const bool produce_;
    const bool mustRunEvent_;
    mutable std::atomic<bool> wasRunEndJob_ = false;
    mutable std::atomic<bool> wasRunEvent_ = false;
  };
}  // namespace edmtest

namespace edm::test {
  namespace other {
    class IntProducer : public edm::global::EDProducer<> {
    public:
      explicit IntProducer(edm::ParameterSet const& p)
          : token_{produces()}, value_(p.getParameter<int>("valueOther")) {}
      void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const final { e.emplace(token_, value_); }

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<int>("valueOther");
        desc.add<int>("valueCpu");
        desc.addUntracked<std::string>("variant", "");

        descriptions.addWithDefaultLabel(desc);
      }

    private:
      edm::EDPutTokenT<edmtest::IntProduct> token_;
      int value_;
    };
  }  // namespace other
  namespace cpu {
    class IntProducer : public edm::global::EDProducer<> {
    public:
      explicit IntProducer(edm::ParameterSet const& p) : token_{produces()}, value_(p.getParameter<int>("valueCpu")) {}
      void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const final { e.emplace(token_, value_); }

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<int>("valueOther");
        desc.add<int>("valueCpu");
        desc.addUntracked<std::string>("variant", "");

        descriptions.addWithDefaultLabel(desc);
      }

    private:
      edm::EDPutTokenT<edmtest::IntProduct> token_;
      int value_;
    };
  }  // namespace cpu
}  // namespace edm::test

using edmtest::AddAllIntsProducer;
using edmtest::AddIntsProducer;
using edmtest::BusyWaitIntLimitedProducer;
using edmtest::BusyWaitIntOneSharedProducer;
using edmtest::BusyWaitIntProducer;
using edmtest::ConsumingIntProducer;
using edmtest::EventNumberIntProducer;
using edmtest::FailingProducer;
using edmtest::Int16_tProducer;
using edmtest::IntOneSharedProducer;
using edmtest::IntProducer;
using edmtest::IntProducerBeginProcessBlock;
using edmtest::IntProducerEndProcessBlock;
using edmtest::IntProducerFromTransient;
using edmtest::ManyIntProducer;
using edmtest::ManyIntWhenRegisteredProducer;
using edmtest::NonEventIntProducer;
using edmtest::NonProducer;
using edmtest::TransientIntProducer;
using edmtest::TransientIntProducerEndProcessBlock;
DEFINE_FWK_MODULE(FailingProducer);
DEFINE_FWK_MODULE(edmtest::FailingInLumiProducer);
DEFINE_FWK_MODULE(edmtest::FailingInRunProducer);
DEFINE_FWK_MODULE(NonProducer);
DEFINE_FWK_MODULE(IntProducer);
DEFINE_FWK_MODULE(IntOneSharedProducer);
DEFINE_FWK_MODULE(BusyWaitIntProducer);
DEFINE_FWK_MODULE(BusyWaitIntLimitedProducer);
DEFINE_FWK_MODULE(BusyWaitIntOneSharedProducer);
DEFINE_FWK_MODULE(ConsumingIntProducer);
DEFINE_FWK_MODULE(EventNumberIntProducer);
DEFINE_FWK_MODULE(TransientIntProducer);
DEFINE_FWK_MODULE(IntProducerFromTransient);
DEFINE_FWK_MODULE(edmtest::TransientIntParentProducer);
DEFINE_FWK_MODULE(edmtest::IntProducerFromTransientParent);
DEFINE_FWK_MODULE(Int16_tProducer);
DEFINE_FWK_MODULE(AddIntsProducer);
DEFINE_FWK_MODULE(AddAllIntsProducer);
DEFINE_FWK_MODULE(ManyIntProducer);
DEFINE_FWK_MODULE(ManyIntWhenRegisteredProducer);
DEFINE_FWK_MODULE(NonEventIntProducer);
DEFINE_FWK_MODULE(IntProducerBeginProcessBlock);
DEFINE_FWK_MODULE(IntProducerEndProcessBlock);
DEFINE_FWK_MODULE(TransientIntProducerEndProcessBlock);
DEFINE_FWK_MODULE(edmtest::MustRunIntProducer);
DEFINE_FWK_MODULE(edm::test::other::IntProducer);
DEFINE_FWK_MODULE(edm::test::cpu::IntProducer);
