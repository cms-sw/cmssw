
/*----------------------------------------------------------------------

Toy EDProducers of Ints for testing purposes only.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
//
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/limited/EDProducer.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
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
  class IntLegacyProducer : public edm::EDProducer {
  public:
    explicit IntLegacyProducer(edm::ParameterSet const& p) : value_(p.getParameter<int>("ivalue")) {
      produces<IntProduct>();
    }
    explicit IntLegacyProducer(int i) : value_(i) { produces<IntProduct>(); }
    void produce(edm::Event& e, edm::EventSetup const& c) override;

  private:
    int value_;
  };

  void IntLegacyProducer::produce(edm::Event& e, edm::EventSetup const&) {
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
  class BusyWaitIntLegacyProducer : public edm::EDProducer {
  public:
    explicit BusyWaitIntLegacyProducer(edm::ParameterSet const& p)
        : value_(p.getParameter<int>("ivalue")),
          iterations_(p.getParameter<unsigned int>("iterations")),
          pi_(std::acos(-1)) {
      produces<IntProduct>();
    }

    void produce(edm::Event& e, edm::EventSetup const& c) override;

  private:
    const int value_;
    const unsigned int iterations_;
    const double pi_;
  };

  void BusyWaitIntLegacyProducer::produce(edm::Event& e, edm::EventSetup const&) {
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
        : token_{produces<IntProduct>()}, value_(p.getParameter<int>("ivalue")) {
      // not used, only exists to test PathAndConsumesOfModules
      consumes<edm::TriggerResults>(edm::InputTag("TriggerResults"));
      consumesMany<edm::TriggerResults>();
    }
    void produce(edm::Event& e, edm::EventSetup const& c) override;

  private:
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
        : putToken_{produces<IntProduct>()}, getToken_{consumes<TransientIntProduct>(edm::InputTag{"TransientThing"})} {}
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
          onlyGetOnEvent_(p.getUntrackedParameter<unsigned int>("onlyGetOnEvent", 0u)) {
      auto const& labels = p.getParameter<std::vector<std::string>>("labels");
      for (auto const& label : labels) {
        tokens_.emplace_back(consumes<IntProduct>(edm::InputTag{label}));
      }
    }
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

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
    bool throw_;
  };

  void ManyIntProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    if (throw_) {
      throw edm::Exception(edm::errors::NotFound) << "Intentional 'NotFound' exception for testing purposes\n";
    }

    // EventSetup is not used.
    for (auto const tv : tokenValues_) {
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
                                                             edm::EndRunProducer> {
  public:
    explicit NonEventIntProducer(edm::ParameterSet const& p)
        : brToken_{produces<IntProduct, edm::Transition::BeginRun>("beginRun")},
          blToken_{produces<IntProduct, edm::Transition::BeginLuminosityBlock>("beginLumi")},
          elToken_{produces<IntProduct, edm::Transition::EndLuminosityBlock>("endLumi")},
          erToken_{produces<IntProduct, edm::Transition::EndRun>("endRun")},
          value_(p.getParameter<int>("ivalue")),
          brExpect_{p.getUntrackedParameter<int>("expectBeginRun")},
          blExpect_{p.getUntrackedParameter<int>("expectBeginLuminosityBlock")},
          elExpect_{p.getUntrackedParameter<int>("expectEndLuminosityBlock")},
          erExpect_{p.getUntrackedParameter<int>("expectEndRun")} {
      {
        auto tag = p.getParameter<edm::InputTag>("consumesBeginRun");
        if (not tag.label().empty()) {
          brGet_ = consumes<IntProduct, edm::InRun>(tag);
        }
      }
      {
        auto tag = p.getParameter<edm::InputTag>("consumesBeginLuminosityBlock");
        if (not tag.label().empty()) {
          blGet_ = consumes<IntProduct, edm::InLumi>(tag);
        }
      }
      {
        auto tag = p.getParameter<edm::InputTag>("consumesEndLuminosityBlock");
        if (not tag.label().empty()) {
          elGet_ = consumes<IntProduct, edm::InLumi>(tag);
        }
      }
      {
        auto tag = p.getParameter<edm::InputTag>("consumesEndRun");
        if (not tag.label().empty()) {
          erGet_ = consumes<IntProduct, edm::InRun>(tag);
        }
      }
    }
    void accumulate(edm::StreamID iID, edm::Event const& e, edm::EventSetup const& c) const override;
    void globalBeginRunProduce(edm::Run& e, edm::EventSetup const&) const override;
    void globalEndRunProduce(edm::Run& e, edm::EventSetup const&) const override;
    void globalBeginLuminosityBlockProduce(edm::LuminosityBlock& e, edm::EventSetup const&) const override;
    void globalEndLuminosityBlockProduce(edm::LuminosityBlock& e, edm::EventSetup const&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& conf) {
      edm::ParameterSetDescription desc;
      desc.add<int>("ivalue", 0);
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
    const edm::EDPutTokenT<IntProduct> brToken_;
    const edm::EDPutTokenT<IntProduct> blToken_;
    const edm::EDPutTokenT<IntProduct> elToken_;
    const edm::EDPutTokenT<IntProduct> erToken_;
    const int value_;
    edm::EDGetTokenT<IntProduct> brGet_;
    edm::EDGetTokenT<IntProduct> blGet_;
    edm::EDGetTokenT<IntProduct> elGet_;
    edm::EDGetTokenT<IntProduct> erGet_;
    const int brExpect_;
    const int blExpect_;
    const int elExpect_;
    const int erExpect_;
  };

  void NonEventIntProducer::accumulate(edm::StreamID iID, edm::Event const& e, edm::EventSetup const&) const {}
  void NonEventIntProducer::globalBeginRunProduce(edm::Run& r, edm::EventSetup const&) const {
    if (not brGet_.isUninitialized()) {
      check(r.get(brGet_), brExpect_);
    }
    r.emplace(brToken_, value_);
  }
  void NonEventIntProducer::globalEndRunProduce(edm::Run& r, edm::EventSetup const&) const {
    if (not erGet_.isUninitialized()) {
      check(r.get(erGet_), erExpect_);
    }
    r.emplace(erToken_, value_);
  }
  void NonEventIntProducer::globalBeginLuminosityBlockProduce(edm::LuminosityBlock& l, edm::EventSetup const&) const {
    if (not blGet_.isUninitialized()) {
      check(l.get(blGet_), blExpect_);
    }
    l.emplace(blToken_, value_);
  }
  void NonEventIntProducer::globalEndLuminosityBlockProduce(edm::LuminosityBlock& l, edm::EventSetup const&) const {
    if (not elGet_.isUninitialized()) {
      check(l.get(elGet_), elExpect_);
    }
    l.emplace(elToken_, value_);
  }
  void NonEventIntProducer::check(IntProduct iProd, int iExpect) const {
    if (iExpect != iProd.value) {
      throw cms::Exception("BadValue") << "expected " << iExpect << " but got " << iProd.value;
    }
  }
}  // namespace edmtest

using edmtest::AddIntsProducer;
using edmtest::BusyWaitIntLegacyProducer;
using edmtest::BusyWaitIntLimitedProducer;
using edmtest::BusyWaitIntProducer;
using edmtest::ConsumingIntProducer;
using edmtest::EventNumberIntProducer;
using edmtest::FailingProducer;
using edmtest::Int16_tProducer;
using edmtest::IntLegacyProducer;
using edmtest::IntProducer;
using edmtest::IntProducerFromTransient;
using edmtest::ManyIntProducer;
using edmtest::ManyIntWhenRegisteredProducer;
using edmtest::NonEventIntProducer;
using edmtest::NonProducer;
using edmtest::TransientIntProducer;
DEFINE_FWK_MODULE(FailingProducer);
DEFINE_FWK_MODULE(edmtest::FailingInRunProducer);
DEFINE_FWK_MODULE(NonProducer);
DEFINE_FWK_MODULE(IntProducer);
DEFINE_FWK_MODULE(IntLegacyProducer);
DEFINE_FWK_MODULE(BusyWaitIntProducer);
DEFINE_FWK_MODULE(BusyWaitIntLimitedProducer);
DEFINE_FWK_MODULE(BusyWaitIntLegacyProducer);
DEFINE_FWK_MODULE(ConsumingIntProducer);
DEFINE_FWK_MODULE(EventNumberIntProducer);
DEFINE_FWK_MODULE(TransientIntProducer);
DEFINE_FWK_MODULE(IntProducerFromTransient);
DEFINE_FWK_MODULE(Int16_tProducer);
DEFINE_FWK_MODULE(AddIntsProducer);
DEFINE_FWK_MODULE(ManyIntProducer);
DEFINE_FWK_MODULE(ManyIntWhenRegisteredProducer);
DEFINE_FWK_MODULE(NonEventIntProducer);
