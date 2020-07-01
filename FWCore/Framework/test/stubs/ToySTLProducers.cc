
/*----------------------------------------------------------------------

Toy EDProducers of STL containers for testing purposes only.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cassert>
#include <string>
#include <vector>

namespace edmtest {

  //--------------------------------------------------------------------
  //
  // STL container producers
  //
  //--------------------------------------------------------------------

  //--------------------------------------------------------------------
  //
  // Produces an std::vector<int> instance.
  //
  class IntVectorProducer : public edm::EDProducer {
  public:
    explicit IntVectorProducer(edm::ParameterSet const& p)
        : value_(p.getParameter<int>("ivalue")),
          count_(p.getParameter<int>("count")),
          delta_(p.getParameter<int>("delta")) {
      produces<std::vector<int>>();
    }
    ~IntVectorProducer() override {}
    void produce(edm::Event& e, edm::EventSetup const& c) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int>("ivalue", 0);
      desc.add<int>("count", 0);
      desc.add<int>("delta", 0);
      descriptions.addDefault(desc);
    }

  private:
    int value_;
    size_t count_;
    int delta_;
  };

  void IntVectorProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    auto p = std::make_unique<std::vector<int>>(count_, value_);
    if (delta_ != 0) {
      for (unsigned int i = 0; i < p->size(); ++i) {
        p->at(i) = value_ + i * delta_;
      }
    }
    e.put(std::move(p));
  }

  //--------------------------------------------------------------------
  //
  // Produces an std::vector<int> and set<int> instance.
  // Used to test ambiguous getByToken calls with View
  // arguments.
  class IntVectorSetProducer : public edm::EDProducer {
  public:
    explicit IntVectorSetProducer(edm::ParameterSet const& p) {
      produces<std::vector<int>>();
      produces<std::set<int>>();
    }
    ~IntVectorSetProducer() override {}
    void produce(edm::Event& e, edm::EventSetup const& c) override;
  };

  void IntVectorSetProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    e.put(std::make_unique<std::vector<int>>(1, 11));

    e.put(std::make_unique<std::set<int>>());
  }

  //--------------------------------------------------------------------
  //
  // Produces an std::list<int> instance.
  //
  class IntListProducer : public edm::EDProducer {
  public:
    explicit IntListProducer(edm::ParameterSet const& p)
        : value_(p.getParameter<int>("ivalue")), count_(p.getParameter<int>("count")) {
      produces<std::list<int>>();
    }
    ~IntListProducer() override {}
    void produce(edm::Event& e, edm::EventSetup const& c) override;

  private:
    int value_;
    size_t count_;
  };

  void IntListProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    e.put(std::make_unique<std::list<int>>(count_, value_));
  }

  //--------------------------------------------------------------------
  //
  // Produces an std::deque<int> instance.
  //
  class IntDequeProducer : public edm::EDProducer {
  public:
    explicit IntDequeProducer(edm::ParameterSet const& p)
        : value_(p.getParameter<int>("ivalue")), count_(p.getParameter<int>("count")) {
      produces<std::deque<int>>();
    }
    ~IntDequeProducer() override {}
    void produce(edm::Event& e, edm::EventSetup const& c) override;

  private:
    int value_;
    size_t count_;
  };

  void IntDequeProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    e.put(std::make_unique<std::deque<int>>(count_, value_));
  }

  //--------------------------------------------------------------------
  //
  // Produces an std::set<int> instance.
  //
  class IntSetProducer : public edm::EDProducer {
  public:
    explicit IntSetProducer(edm::ParameterSet const& p)
        : start_(p.getParameter<int>("start")), stop_(p.getParameter<int>("stop")) {
      produces<std::set<int>>();
    }
    ~IntSetProducer() override {}
    void produce(edm::Event& e, edm::EventSetup const& c) override;

  private:
    int start_;
    int stop_;
  };

  void IntSetProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    auto p = std::make_unique<std::set<int>>();
    for (int i = start_; i < stop_; ++i)
      p->insert(i);
    e.put(std::move(p));
  }

  //--------------------------------------------------------------------
  //
  // Produces std::vector<std::unique_ptr<int>> and std::vector<std::unique_ptr<IntProduct>> instances.
  //
  class UniqPtrIntVectorProducer : public edm::EDProducer {
  public:
    explicit UniqPtrIntVectorProducer(edm::ParameterSet const& p)
        : value_(p.getParameter<int>("ivalue")),
          count_(p.getParameter<int>("count")),
          delta_(p.getParameter<int>("delta")),
          intToken_(produces<std::vector<std::unique_ptr<int>>>()),
          intProductToken_(produces<std::vector<std::unique_ptr<IntProduct>>>()) {}

    void produce(edm::Event& e, edm::EventSetup const& c) override {
      std::vector<std::unique_ptr<int>> p1(count_);
      std::vector<std::unique_ptr<IntProduct>> p2(count_);
      for (unsigned int i = 0; i < count_; ++i) {
        const int v = value_ + i * delta_;
        p1[i] = std::make_unique<int>(v);
        p2[i] = std::make_unique<IntProduct>(v);
      }
      e.emplace(intToken_, std::move(p1));
      e.emplace(intProductToken_, std::move(p2));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int>("ivalue", 0);
      desc.add<int>("count", 0);
      desc.add<int>("delta", 0);
      descriptions.addDefault(desc);
    }

  private:
    int value_;
    size_t count_;
    int delta_;
    edm::EDPutTokenT<std::vector<std::unique_ptr<int>>> intToken_;
    edm::EDPutTokenT<std::vector<std::unique_ptr<IntProduct>>> intProductToken_;
  };
}  // namespace edmtest

using edmtest::IntDequeProducer;
using edmtest::IntListProducer;
using edmtest::IntSetProducer;
using edmtest::IntVectorProducer;
using edmtest::IntVectorSetProducer;
using edmtest::UniqPtrIntVectorProducer;
DEFINE_FWK_MODULE(IntVectorProducer);
DEFINE_FWK_MODULE(IntVectorSetProducer);
DEFINE_FWK_MODULE(IntListProducer);
DEFINE_FWK_MODULE(IntDequeProducer);
DEFINE_FWK_MODULE(IntSetProducer);
DEFINE_FWK_MODULE(UniqPtrIntVectorProducer);
