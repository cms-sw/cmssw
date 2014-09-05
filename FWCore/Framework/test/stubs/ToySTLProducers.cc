
/*----------------------------------------------------------------------

Toy EDProducers of STL containers for testing purposes only.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "FWCore/Framework/interface/EDProducer.h"
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
    explicit IntVectorProducer(edm::ParameterSet const& p) :
      value_(p.getParameter<int>("ivalue")),
      count_(p.getParameter<int>("count")) {
      produces<std::vector<int> >();
    }
    virtual ~IntVectorProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);
  private:
    int    value_;
    size_t count_;
  };

  void
  IntVectorProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    std::unique_ptr<std::vector<int> > p(new std::vector<int>(count_, value_));
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
      produces<std::vector<int> >();
      produces<std::set<int> >();
    }
    virtual ~IntVectorSetProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);
  };

  void
  IntVectorSetProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    std::unique_ptr<std::vector<int> > p(new std::vector<int>(1,11));
    e.put(std::move(p));

    std::unique_ptr<std::set<int> > apset(new std::set<int>);
    e.put(std::move(apset));
  }

  //--------------------------------------------------------------------
  //
  // Produces an std::list<int> instance.
  //
  class IntListProducer : public edm::EDProducer {
  public:
    explicit IntListProducer(edm::ParameterSet const& p) :
        value_(p.getParameter<int>("ivalue")),
        count_(p.getParameter<int>("count")) {
      produces<std::list<int> >();
    }
    virtual ~IntListProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);
  private:
    int    value_;
    size_t count_;
  };

  void
  IntListProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    std::unique_ptr<std::list<int> > p(new std::list<int>(count_, value_));
    e.put(std::move(p));
  }

  //--------------------------------------------------------------------
  //
  // Produces an std::deque<int> instance.
  //
  class IntDequeProducer : public edm::EDProducer {
  public:
    explicit IntDequeProducer(edm::ParameterSet const& p) :
        value_(p.getParameter<int>("ivalue")),
        count_(p.getParameter<int>("count")) {
      produces<std::deque<int> >();
    }
    virtual ~IntDequeProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);
  private:
    int    value_;
    size_t count_;
  };

  void
  IntDequeProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    std::unique_ptr<std::deque<int> > p(new std::deque<int>(count_, value_));
    e.put(std::move(p));
  }

  //--------------------------------------------------------------------
  //
  // Produces an std::set<int> instance.
  //
  class IntSetProducer : public edm::EDProducer {
  public:
    explicit IntSetProducer(edm::ParameterSet const& p) :
        start_(p.getParameter<int>("start")),
        stop_(p.getParameter<int>("stop")) {
      produces<std::set<int> >();
    }
    virtual ~IntSetProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);
  private:
    int start_;
    int stop_;
  };

  void
  IntSetProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    std::unique_ptr<std::set<int> > p(new std::set<int>());
    for(int i = start_; i < stop_; ++i) p->insert(i);
    e.put(std::move(p));
  }

}

using edmtest::IntVectorProducer;
using edmtest::IntVectorSetProducer;
using edmtest::IntListProducer;
using edmtest::IntDequeProducer;
using edmtest::IntSetProducer;
DEFINE_FWK_MODULE(IntVectorProducer);
DEFINE_FWK_MODULE(IntVectorSetProducer);
DEFINE_FWK_MODULE(IntListProducer);
DEFINE_FWK_MODULE(IntDequeProducer);
DEFINE_FWK_MODULE(IntSetProducer);

