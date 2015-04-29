
/*----------------------------------------------------------------------

Toy EDProducers of Ref/Ptr for testing purposes only.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <cassert>
#include <string>
#include <vector>

namespace edmtest {

  //--------------------------------------------------------------------
  //
  // Toy Ref or Ptr producers
  //
  //--------------------------------------------------------------------


  //--------------------------------------------------------------------
  //
  // Produces an edm::RefVector<std::vector<int> > instance.
  // This requires that an instance of IntVectorProducer be run *before*
  // this producer.
  class IntVecRefVectorProducer : public edm::global::EDProducer<> {
    typedef edm::RefVector<std::vector<int> > product_type;

  public:
    explicit IntVecRefVectorProducer(edm::ParameterSet const& p) :
        target_{consumes<std::vector<int>>(p.getParameter<edm::InputTag>("target"))},
        select_(p.getParameter<int>("select")) {
      produces<product_type>();
    }
    virtual void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int>("select", 0);
      desc.add<edm::InputTag>("target");
      descriptions.addDefault(desc);
    }

  private:
    const edm::EDGetTokenT<std::vector<int>> target_;
    int select_;
  };

  void
  IntVecRefVectorProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    // EventSetup is not used.
    // Get our input:
    edm::Handle<std::vector<int> > input;
    e.getByToken(target_, input);
    assert(input.isValid());

    std::unique_ptr<product_type> prod(new product_type());

    typedef product_type::value_type ref;
    for(size_t i = 0, sz = input->size(); i != sz; ++i) {
      if(select_ != 0 && (i % select_) == 0) continue;
      prod->push_back(ref(input, i));
    }

    e.put(std::move(prod));
  }

  //--------------------------------------------------------------------
  //
  // Produces an edm::RefToBaseVector<int> instance.
  // This requires that an instance of IntVectorProducer be run *before*
  // this producer. The input collection is read as an edm::View<int>
  class IntVecRefToBaseVectorProducer : public edm::global::EDProducer<> {
    typedef edm::RefToBaseVector<int> product_type;

  public:
    explicit IntVecRefToBaseVectorProducer(edm::ParameterSet const& p) :
      target_{consumes<edm::View<int>>(p.getParameter<edm::InputTag>("target"))} {
      produces<product_type>();
    }
    virtual void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

  private:
    const edm::EDGetTokenT<edm::View<int>> target_;
  };

  void
  IntVecRefToBaseVectorProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    // EventSetup is not used.
    // Get our input:
    edm::Handle<edm::View<int> > input;
    e.getByToken(target_, input);
    assert(input.isValid());

    edm::RefToBaseVector<int> refVector;
    for (size_t i = 0; i < input->size(); ++i) {
      refVector.push_back(input->refAt(i));
    }

    std::unique_ptr<product_type> prod(new product_type(refVector));
    e.put(std::move(prod));
  }

  //--------------------------------------------------------------------
  //
  // Produces an edm::PtrVector<int> instance.
  // This requires that an instance of IntVectorProducer be run *before*
  // this producer. The input collection is read as an edm::View<int>
  class IntVecPtrVectorProducer : public edm::global::EDProducer<> {
    typedef edm::PtrVector<int> product_type;

  public:
    explicit IntVecPtrVectorProducer(edm::ParameterSet const& p) :
    target_{consumes<edm::View<int>>(p.getParameter<edm::InputTag>("target"))} {
      produces<product_type>();
    }
    virtual void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

  private:
    const edm::EDGetTokenT<edm::View<int>> target_;
  };

  void
  IntVecPtrVectorProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    // EventSetup is not used.
    // Get our input:
    edm::Handle<edm::View<int> > input;
    e.getByToken(target_, input);
    assert(input.isValid());

    std::unique_ptr<product_type> prod(new product_type());

    typedef product_type::value_type ref;
    for(size_t i = 0, sz = input->size(); i != sz; ++i)
      prod->push_back(ref(input, i));

    e.put(std::move(prod));
  }

  //--------------------------------------------------------------------
  //
  // Produces an std::vector<edm::Ptr<int>> instance.
  // This requires that an instance of IntVectorProducer be run *before*
  // this producer. The input collection is read as an edm::View<int>
  class IntVecStdVectorPtrProducer : public edm::global::EDProducer<> {
    typedef std::vector<edm::Ptr<int>> product_type;

  public:
    explicit IntVecStdVectorPtrProducer(edm::ParameterSet const& p) :
    target_(consumes<edm::View<int>>(p.getParameter<edm::InputTag>("target"))) {
      produces<product_type>();
    }
    virtual void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

  private:
    const edm::EDGetTokenT<edm::View<int>> target_;
  };

  void
  IntVecStdVectorPtrProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const  {
    // EventSetup is not used.
    // Get our input:
    edm::Handle<edm::View<int> > input;
    e.getByToken(target_, input);
    assert(input.isValid());

    std::unique_ptr<product_type> prod(new product_type());

    typedef product_type::value_type ref;
    for(size_t i = 0, sz = input->size(); i != sz; ++i)
      prod->emplace_back(input, i);

    e.put(std::move(prod));
  }

}

using edmtest::IntVecRefVectorProducer;
using edmtest::IntVecRefToBaseVectorProducer;
using edmtest::IntVecPtrVectorProducer;
using edmtest::IntVecStdVectorPtrProducer;
DEFINE_FWK_MODULE(IntVecRefVectorProducer);
DEFINE_FWK_MODULE(IntVecRefToBaseVectorProducer);
DEFINE_FWK_MODULE(IntVecPtrVectorProducer);
DEFINE_FWK_MODULE(IntVecStdVectorPtrProducer);

