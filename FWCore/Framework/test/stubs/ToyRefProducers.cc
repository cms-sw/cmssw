
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
  // Toy Ref or Ptr producers
  //
  //--------------------------------------------------------------------


  //--------------------------------------------------------------------
  //
  // Produces an edm::RefVector<std::vector<int> > instance.
  // This requires that an instance of IntVectorProducer be run *before*
  // this producer.
  class IntVecRefVectorProducer : public edm::EDProducer {
    typedef edm::RefVector<std::vector<int> > product_type;

  public:
    explicit IntVecRefVectorProducer(edm::ParameterSet const& p) :
        target_(p.getParameter<std::string>("target")) {
      produces<product_type>();
      consumes<std::vector<int>>(edm::InputTag{target_});
    }
    virtual ~IntVecRefVectorProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    std::string target_;
  };

  void
  IntVecRefVectorProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    // Get our input:
    edm::Handle<std::vector<int> > input;
    e.getByLabel(target_, input);
    assert(input.isValid());

    std::unique_ptr<product_type> prod(new product_type());

    typedef product_type::value_type ref;
    for(size_t i = 0, sz = input->size(); i != sz; ++i)
      prod->push_back(ref(input, i));

    e.put(std::move(prod));
  }

  //--------------------------------------------------------------------
  //
  // Produces an edm::RefToBaseVector<int> instance.
  // This requires that an instance of IntVectorProducer be run *before*
  // this producer. The input collection is read as an edm::View<int>
  class IntVecRefToBaseVectorProducer : public edm::EDProducer {
    typedef edm::RefToBaseVector<int> product_type;

  public:
    explicit IntVecRefToBaseVectorProducer(edm::ParameterSet const& p) :
        target_(p.getParameter<std::string>("target")) {
      produces<product_type>();
      consumes<edm::View<int>>(edm::InputTag{target_});
    }
    virtual ~IntVecRefToBaseVectorProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    std::string target_;
  };

  void
  IntVecRefToBaseVectorProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    // Get our input:
    edm::Handle<edm::View<int> > input;
    e.getByLabel(target_, input);
    assert(input.isValid());

    std::unique_ptr<product_type> prod(new product_type(input->refVector()));
    e.put(std::move(prod));
  }

  //--------------------------------------------------------------------
  //
  // Produces an edm::PtrVector<int> instance.
  // This requires that an instance of IntVectorProducer be run *before*
  // this producer. The input collection is read as an edm::View<int>
  class IntVecPtrVectorProducer : public edm::EDProducer {
    typedef edm::PtrVector<int> product_type;

  public:
    explicit IntVecPtrVectorProducer(edm::ParameterSet const& p) :
        target_(p.getParameter<std::string>("target")) {
      produces<product_type>();
      consumes<edm::View<int>>(edm::InputTag{target_});
    }
    virtual ~IntVecPtrVectorProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    std::string target_;
  };

  void
  IntVecPtrVectorProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    // Get our input:
    edm::Handle<edm::View<int> > input;
    e.getByLabel(target_, input);
    assert(input.isValid());

    std::unique_ptr<product_type> prod(new product_type());

    typedef product_type::value_type ref;
    for(size_t i = 0, sz = input->size(); i != sz; ++i)
      prod->push_back(ref(input, i));

    e.put(std::move(prod));
  }

}

using edmtest::IntVecRefVectorProducer;
using edmtest::IntVecRefToBaseVectorProducer;
using edmtest::IntVecPtrVectorProducer;
DEFINE_FWK_MODULE(IntVecRefVectorProducer);
DEFINE_FWK_MODULE(IntVecRefToBaseVectorProducer);
DEFINE_FWK_MODULE(IntVecPtrVectorProducer);

