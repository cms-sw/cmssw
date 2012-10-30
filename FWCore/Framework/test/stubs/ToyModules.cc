
/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace edmtest {

  //--------------------------------------------------------------------
  //
  // Toy producers
  //
  //--------------------------------------------------------------------

  //--------------------------------------------------------------------
  //
  // throws an exception.
  // Announces an IntProduct but does not produce one;
  // every call to FailingProducer::produce throws a cms exception
  //
  class FailingProducer : public edm::EDProducer {
  public:
    explicit FailingProducer(edm::ParameterSet const& /*p*/) {
      produces<IntProduct>();
    }
    virtual ~FailingProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);
  };

  void
  FailingProducer::produce(edm::Event&, edm::EventSetup const&) {
    // We throw an edm exception with a configurable action.
    throw edm::Exception(edm::errors::NotFound) << "Intentional 'NotFound' exception for testing purposes\n";
  }

  //--------------------------------------------------------------------
  //
  // Announces an IntProduct but does not produce one;
  // every call to NonProducer::produce does nothing.
  //
  class NonProducer : public edm::EDProducer {
  public:
    explicit NonProducer(edm::ParameterSet const& /*p*/) {
      produces<IntProduct>();
    }
    virtual ~NonProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);
  };

  void
  NonProducer::produce(edm::Event&, edm::EventSetup const&) {
  }

  //--------------------------------------------------------------------
  //
  // every call to NonAnalyzer::analyze does nothing.
  //
  class NonAnalyzer : public edm::EDAnalyzer {
  public:
    explicit NonAnalyzer(edm::ParameterSet const& /*p*/) {
    }
    virtual ~NonAnalyzer() {}
    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  };

  void
  NonAnalyzer::analyze(edm::Event const&, edm::EventSetup const&) {
  }

  //--------------------------------------------------------------------
  //
  // Produces an IntProduct instance.
  //
  class IntProducer : public edm::EDProducer {
  public:
    explicit IntProducer(edm::ParameterSet const& p) :
      value_(p.getParameter<int>("ivalue")) {
      produces<IntProduct>();
    }
    explicit IntProducer(int i) : value_(i) {
      produces<IntProduct>();
    }
    virtual ~IntProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    int value_;
  };

  void
  IntProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    std::auto_ptr<IntProduct> p(new IntProduct(value_));
    e.put(p);
  }

  //--------------------------------------------------------------------
  //
  // Produces an IntProduct instance whose value is the event number,
  // rather than the value of a configured parameter.
  //
  class EventNumberIntProducer : public edm::EDProducer {
  public:
    explicit EventNumberIntProducer(edm::ParameterSet const&) {
      produces<IntProduct>();
    }
    EventNumberIntProducer() {
      produces<IntProduct>();
    }
    virtual ~EventNumberIntProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
  };

  void
  EventNumberIntProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    std::auto_ptr<IntProduct> p(new IntProduct(e.id().event()));
    e.put(p);
  }


  //--------------------------------------------------------------------
  //
  // Produces a TransientIntProduct instance.
  //
  class TransientIntProducer : public edm::EDProducer {
  public:
    explicit TransientIntProducer(edm::ParameterSet const& p) :
      value_(p.getParameter<int>("ivalue")) {
      produces<TransientIntProduct>();
    }
    explicit TransientIntProducer(int i) : value_(i) {
      produces<TransientIntProduct>();
    }
    virtual ~TransientIntProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    int value_;
  };

  void
  TransientIntProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    std::auto_ptr<TransientIntProduct> p(new TransientIntProduct(value_));
    e.put(p);
  }

  //--------------------------------------------------------------------
  //
  // Produces a IntProduct instance from a TransientIntProduct
  //
  class IntProducerFromTransient : public edm::EDProducer {
  public:
    explicit IntProducerFromTransient(edm::ParameterSet const&) {
      produces<IntProduct>();
    }
    explicit IntProducerFromTransient() {
      produces<IntProduct>();
    }
    virtual ~IntProducerFromTransient() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
  };

  void
  IntProducerFromTransient::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    edm::Handle<TransientIntProduct> result;
    bool ok = e.getByLabel("TransientThing", result);
    assert(ok);
    std::auto_ptr<IntProduct> p(new IntProduct(result.product()->value));
    e.put(p);
  }

  //--------------------------------------------------------------------
  //
  // Produces an Int16_tProduct instance.
  //
  class Int16_tProducer : public edm::EDProducer {
  public:
    explicit Int16_tProducer(edm::ParameterSet const& p) :
      value_(p.getParameter<int>("ivalue")) {
      produces<Int16_tProduct>();
    }
    explicit Int16_tProducer(boost::int16_t i, boost::uint16_t) : value_(i) {
      produces<Int16_tProduct>();
    }
    virtual ~Int16_tProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    boost::int16_t value_;
  };

  void
  Int16_tProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    std::auto_ptr<Int16_tProduct> p(new Int16_tProduct(value_));
    e.put(p);
  }

  //--------------------------------------------------------------------
  //
  // Produces an DoubleProduct instance.
  //

  class ToyDoubleProducer : public edm::EDProducer {
  public:
    explicit ToyDoubleProducer(edm::ParameterSet const& p) :
      value_(p.getParameter<double>("dvalue")) {
      produces<DoubleProduct>();
    }
    explicit ToyDoubleProducer(double d) : value_(d) {
      produces<DoubleProduct>();
    }
    virtual ~ToyDoubleProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);
  private:
    double value_;
  };

  void
  ToyDoubleProducer::produce(edm::Event& e, edm::EventSetup const&) {

    // Make output
    std::auto_ptr<DoubleProduct> p(new DoubleProduct(value_));
    e.put(p);
  }

  //--------------------------------------------------------------------
  //
  // Produces an IntProduct instance, using an IntProduct as input.
  //

  class AddIntsProducer : public edm::EDProducer {
  public:
    explicit AddIntsProducer(edm::ParameterSet const& p) :
        labels_(p.getParameter<std::vector<std::string> >("labels")) {
      produces<IntProduct>();
    }
    virtual ~AddIntsProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);
  private:
    std::vector<std::string> labels_;
  };

  void
  AddIntsProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    int value = 0;
    for(std::vector<std::string>::iterator itLabel = labels_.begin(), itLabelEnd = labels_.end();
        itLabel != itLabelEnd; ++itLabel) {
      edm::Handle<IntProduct> anInt;
      e.getByLabel(*itLabel, anInt);
      value += anInt->value;
    }
    std::auto_ptr<IntProduct> p(new IntProduct(value));
    e.put(p);
  }

  //--------------------------------------------------------------------
  //
  // Produces and SCSimpleProduct product instance.
  //
  class SCSimpleProducer : public edm::EDProducer {
  public:
    explicit SCSimpleProducer(edm::ParameterSet const& p) :
      size_(p.getParameter<int>("size")) {
      produces<SCSimpleProduct>();
      assert (size_ > 1);
    }

    explicit SCSimpleProducer(int i) : size_(i) {
      produces<SCSimpleProduct>();
      assert (size_ > 1);
    }

    virtual ~SCSimpleProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    int size_;  // number of Simples to put in the collection
  };

  void
  SCSimpleProducer::produce(edm::Event& e,
                            edm::EventSetup const& /* unused */) {
    // Fill up a collection so that it is sorted *backwards*.
    std::vector<Simple> guts(size_);
    for(int i = 0; i < size_; ++i) {
        guts[i].key = size_ - i;
        guts[i].value = 1.5 * i;
    }

    // Verify that the vector is not sorted -- in fact, it is sorted
    // backwards!
    for(int i = 1; i < size_; ++i) {
        assert(guts[i-1].id() > guts[i].id());
    }

    std::auto_ptr<SCSimpleProduct> p(new SCSimpleProduct(guts));

    // Put the product into the Event, thus sorting it.
    e.put(p);
  }

  //--------------------------------------------------------------------
  //
  // Produces and OVSimpleProduct product instance.
  //
  class OVSimpleProducer : public edm::EDProducer {
  public:
    explicit OVSimpleProducer(edm::ParameterSet const& p) :
        size_(p.getParameter<int>("size")) {
      produces<OVSimpleProduct>();
      produces<OVSimpleDerivedProduct>("derived");
      assert (size_ > 1);
    }

    explicit OVSimpleProducer(int i) : size_(i) {
      produces<OVSimpleProduct>();
      produces<OVSimpleDerivedProduct>("derived");
      assert (size_ > 1);
    }

    virtual ~OVSimpleProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    int size_;  // number of Simples to put in the collection
  };

  void
  OVSimpleProducer::produce(edm::Event& e,
                            edm::EventSetup const& /* unused */) {
    // Fill up a collection
    std::auto_ptr<OVSimpleProduct> p(new OVSimpleProduct());

    for(int i = 0; i < size_; ++i) {
        std::auto_ptr<Simple> simple(new Simple());
        simple->key = size_ - i;
        simple->value = 1.5 * i;
        p->push_back(simple);
    }

    // Put the product into the Event
    e.put(p);

    // Fill up a collection of SimpleDerived objects
    std::auto_ptr<OVSimpleDerivedProduct> pd(new OVSimpleDerivedProduct());

    for(int i = 0; i < size_; ++i) {
        std::auto_ptr<SimpleDerived> simpleDerived(new SimpleDerived());
        simpleDerived->key = size_ - i;
        simpleDerived->value = 1.5 * i + 100.0;
        simpleDerived->dummy = 0.0;
        pd->push_back(simpleDerived);
    }

    // Put the product into the Event
    e.put(pd, "derived");
  }

  //--------------------------------------------------------------------
  //
  // Produces and OVSimpleProduct product instance.
  //
  class VSimpleProducer : public edm::EDProducer {
  public:
    explicit VSimpleProducer(edm::ParameterSet const& p) :
        size_(p.getParameter<int>("size")) {
      produces<VSimpleProduct>();
      assert (size_ > 1);
    }

    explicit VSimpleProducer(int i) : size_(i) {
      produces<VSimpleProduct>();
      assert (size_ > 1);
    }

    virtual ~VSimpleProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    int size_;  // number of Simples to put in the collection
  };

  void
  VSimpleProducer::produce(edm::Event& e,
                           edm::EventSetup const& /* unused */) {
    // Fill up a collection
    std::auto_ptr<VSimpleProduct> p(new VSimpleProduct());

    for(int i = 0; i < size_; ++i) {
        Simple simple;
        simple.key = size_ - i;
        simple.value = 1.5 * i;
        p->push_back(simple);
    }

    // Put the product into the Event
    e.put(p);
  }

  //--------------------------------------------------------------------
  //
  // Produces AssociationVector<vector<Simple>, vector<Simple> > object
  // This is used to test a View of an AssociationVector
  //
  class AVSimpleProducer : public edm::EDProducer {
  public:

    explicit AVSimpleProducer(edm::ParameterSet const& p) :
    src_(p.getParameter<edm::InputTag>("src")) {
      produces<AVSimpleProduct>();
    }

    virtual ~AVSimpleProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    edm::InputTag src_;
  };

  void
  AVSimpleProducer::produce(edm::Event& e,
                            edm::EventSetup const& /* unused */) {
    edm::Handle<std::vector<edmtest::Simple> > vs;
    e.getByLabel(src_, vs);
    // Fill up a collection
    std::auto_ptr<AVSimpleProduct> p(new AVSimpleProduct(edm::RefProd<std::vector<edmtest::Simple> >(vs)));

    for(unsigned int i = 0; i < vs->size(); ++i) {
        edmtest::Simple simple;
        simple.key = 100 + i;  // just some arbitrary number for testing
        p->setValue(i, simple);
    }

    // Put the product into the Event
    e.put(p);
  }

  //--------------------------------------------------------------------
  //
  // Produces two products:
  //    DSVSimpleProduct
  //    DSVWeirdProduct
  //
  class DSVProducer : public edm::EDProducer {
  public:

    explicit DSVProducer(edm::ParameterSet const& p) :
        size_(p.getParameter<int>("size")) {
      produces<DSVSimpleProduct>();
      produces<DSVWeirdProduct>();
      assert(size_ > 1);
    }

    explicit DSVProducer(int i) : size_(i) {
      produces<DSVSimpleProduct>();
      produces<DSVWeirdProduct>();
      assert(size_ > 1);
    }

    virtual ~DSVProducer() {}

    virtual void produce(edm::Event& e, edm::EventSetup const&);

  private:
    template<typename PROD> void make_a_product(edm::Event& e);
    int size_;
  };

  void
  DSVProducer::produce(edm::Event& e, edm::EventSetup const& /* unused */) {
    this->make_a_product<DSVSimpleProduct>(e);
    this->make_a_product<DSVWeirdProduct>(e);
  }

  template<typename PROD>
  void
  DSVProducer::make_a_product(edm::Event& e) {
    typedef PROD                              product_type;
    typedef typename product_type::value_type detset;
    typedef typename detset::value_type       value_type;

    // Fill up a collection so that it is sorted *backwards*.
    std::vector<value_type> guts(size_);
    for(int i = 0; i < size_; ++i) {
      guts[i].data = size_ - i;
    }

    // Verify that the vector is not sorted -- in fact, it is sorted
    // backwards!
    for(int i = 1; i < size_; ++i) {
      assert(guts[i-1].data > guts[i].data);
    }

    std::auto_ptr<product_type> p(new product_type());
    int n = 0;
    for(int id = 1; id<size_; ++id) {
      ++n;
      detset item(id); // this will get DetID id
      item.data.insert(item.data.end(), guts.begin(), guts.begin()+n);
      p->insert(item);
    }

    // Put the product into the Event, thus sorting it ... or not,
    // depending upon the product type.
    e.put(p);
  }

  //--------------------------------------------------------------------
  //
  // Produces two products: (new DataSetVector)
  //    DSTVSimpleProduct
  //    DSTVSimpleDerivedProduct
  //
  class DSTVProducer : public edm::EDProducer {
  public:

    explicit DSTVProducer(edm::ParameterSet const& p) :
        size_(p.getParameter<int>("size")) {
      produces<DSTVSimpleProduct>();
      produces<DSTVSimpleDerivedProduct>();
      assert(size_ > 1);
    }

    explicit DSTVProducer(int i) : size_(i) {
      produces<DSTVSimpleProduct>();
      produces<DSTVSimpleDerivedProduct>();
      assert(size_ > 1);
    }

    virtual ~DSTVProducer() {}

    virtual void produce(edm::Event& e, edm::EventSetup const&);

  private:
    template<typename PROD> void make_a_product(edm::Event& e);
    void fill_a_data(DSTVSimpleProduct::data_type& d, unsigned int i);
    void fill_a_data(DSTVSimpleDerivedProduct::data_type& d, unsigned int i);

    int size_;
  };

  void
  DSTVProducer::produce(edm::Event& e, edm::EventSetup const& /* unused */) {
    this->make_a_product<DSTVSimpleProduct>(e);
    this->make_a_product<DSTVSimpleDerivedProduct>(e);
  }

  void
  DSTVProducer::fill_a_data(DSTVSimpleDerivedProduct::data_type& d, unsigned int i) {
    d.key = size_ - i;
    d.value = 1.5 * i;
  }

  void
  DSTVProducer::fill_a_data(DSTVSimpleProduct::data_type& d, unsigned int i) {
    d.data = size_ - i;
  }

  template<typename PROD>
  void
  DSTVProducer::make_a_product(edm::Event& e) {
    typedef PROD                     product_type;
    //FIXME
    typedef typename product_type::FastFiller detset;
    typedef typename detset::value_type       value_type;
    typedef typename detset::id_type       id_type;

    std::auto_ptr<product_type> p(new product_type());
    product_type& v = *p;

    unsigned int n = 0;
    for(id_type id = 1; id<static_cast<id_type>(size_) ;++id) {
      ++n;
      detset item(v, id); // this will get DetID id
      item.resize(n);
      for(unsigned int i = 0; i < n; ++i)
        fill_a_data(item[i], i);
    }

    // Put the product into the Event, thus sorting is not done by magic,
    // up to one user-line
    e.put(p);
  }

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
    std::auto_ptr<std::vector<int> > p(new std::vector<int>(count_, value_));
    e.put(p);
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
    std::auto_ptr<std::list<int> > p(new std::list<int>(count_, value_));
    e.put(p);
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
    std::auto_ptr<std::deque<int> > p(new std::deque<int>(count_, value_));
    e.put(p);
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
    std::auto_ptr<std::set<int> > p(new std::set<int>());
    for(int i = start_; i < stop_; ++i) p->insert(i);
    e.put(p);
  }

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

    std::auto_ptr<product_type> prod(new product_type());

    typedef product_type::value_type ref;
    for(size_t i = 0, sz = input->size(); i != sz; ++i)
      prod->push_back(ref(input, i));

    e.put(prod);
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

    std::auto_ptr<product_type> prod(new product_type(input->refVector()));
    e.put(prod);
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

    std::auto_ptr<product_type> prod(new product_type());

    typedef product_type::value_type ref;
    for(size_t i = 0, sz = input->size(); i != sz; ++i)
      prod->push_back(ref(input, i));

    e.put(prod);
  }

  //--------------------------------------------------------------------
  //
  // Produces an Prodigal instance.
  //
  class ProdigalProducer : public edm::EDProducer {
  public:
    explicit ProdigalProducer(edm::ParameterSet const& p) :
      label_(p.getParameter<std::string>("label")) {
      produces<Prodigal>();
    }
    virtual ~ProdigalProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    std::string label_;
  };

  void
  ProdigalProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.

    // The purpose of Prodigal is testing of *not* keeping
    // parentage. So we need to get a product...
    edm::Handle<IntProduct> parent;
    e.getByLabel(label_, parent);

    std::auto_ptr<Prodigal> p(new Prodigal(parent->value));
    e.put(p);
  }

  //--------------------------------------------------------------------
  //
  // Toy analyzers
  //
  //--------------------------------------------------------------------

  //--------------------------------------------------------------------
  //
  class IntTestAnalyzer : public edm::EDAnalyzer {
  public:
    IntTestAnalyzer(edm::ParameterSet const& iPSet) :
      value_(iPSet.getUntrackedParameter<int>("valueMustMatch")),
      moduleLabel_(iPSet.getUntrackedParameter<std::string>("moduleLabel"), "") {
    }

    void analyze(edm::Event const& iEvent, edm::EventSetup const&) {
      edm::Handle<IntProduct> handle;
      iEvent.getByLabel(moduleLabel_, handle);
      if(handle->value != value_) {
        throw cms::Exception("ValueMissMatch")
          << "The value for \"" << moduleLabel_ << "\" is "
          << handle->value << " but it was supposed to be " << value_;
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

    virtual void
    analyze(edm::Event const& e, edm::EventSetup const&);
  };

  void
  SCSimpleAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&) {

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

    for(size_type i = 1, end = after.size(); i < end; ++i) {
      assert(after[i-1].id() < after[i].id());
    }
  }

  //--------------------------------------------------------------------
  //
  class DSVAnalyzer : public edm::EDAnalyzer {
  public:
    DSVAnalyzer(edm::ParameterSet const&) {}

    virtual void
    analyze(edm::Event const& e, edm::EventSetup const&);
  private:
    void do_sorted_stuff(edm::Event const& e);
    void do_unsorted_stuff(edm::Event const& e);
  };

  void
  DSVAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&) {
    do_sorted_stuff(e);
    do_unsorted_stuff(e);
  }

  void
  DSVAnalyzer::do_sorted_stuff(edm::Event const& e) {
    typedef DSVSimpleProduct         product_type;
    typedef product_type::value_type detset;
    typedef detset::value_type       value_type;
    // Get the product back out; it should be sorted.
    edm::Handle<product_type> h;
    e.getByLabel("dsv1", h);
    assert(h.isValid());

    // Check the sorting. DO NOT DO THIS IN NORMAL CODE; we are
    // copying all the values out of the DetSetVector's first DetSet so we can
    // manipulate them via an interface different from
    // DetSet, just so that we can make sure the collection
    // is sorted.
    std::vector<value_type> const& after = (h->end()-1)->data;
    typedef std::vector<value_type>::size_type size_type;

    // Verify that the vector *is* sorted.

    for(size_type i = 1, end = after.size(); i < end; ++i) {
      assert(after[i-1].data < after[i].data);
    }
  }

  void
  DSVAnalyzer::do_unsorted_stuff(edm::Event const& e) {
    typedef DSVWeirdProduct         product_type;
    typedef product_type::value_type detset;
    typedef detset::value_type       value_type;
    // Get the product back out; it should be unsorted.
    edm::Handle<product_type> h;
    e.getByLabel("dsv1", h);
    assert(h.isValid());

    // Check the sorting. DO NOT DO THIS IN NORMAL CODE; we are
    // copying all the values out of the DetSetVector's first DetSet so we can
    // manipulate them via an interface different from
    // DetSet, just so that we can make sure the collection
    // is not sorted.
    std::vector<value_type> const& after = (h->end()-1)->data;
    typedef std::vector<value_type>::size_type size_type;

    // Verify that the vector is reverse-sorted.

    for(size_type i = 1, end = after.size(); i < end; ++i) {
        assert(after[i-1].data > after[i].data);
    }
  }
}

using edmtest::FailingProducer;
using edmtest::NonAnalyzer;
using edmtest::NonProducer;
using edmtest::IntProducer;
using edmtest::EventNumberIntProducer;
using edmtest::TransientIntProducer;
using edmtest::IntProducerFromTransient;
using edmtest::Int16_tProducer;
using edmtest::ToyDoubleProducer;
using edmtest::SCSimpleProducer;
using edmtest::OVSimpleProducer;
using edmtest::VSimpleProducer;
using edmtest::AVSimpleProducer;
using edmtest::DSTVProducer;
using edmtest::DSVProducer;
using edmtest::IntTestAnalyzer;
using edmtest::SCSimpleAnalyzer;
using edmtest::DSVAnalyzer;
using edmtest::AddIntsProducer;
using edmtest::IntVectorProducer;
using edmtest::IntListProducer;
using edmtest::IntDequeProducer;
using edmtest::IntSetProducer;
using edmtest::IntVecRefVectorProducer;
using edmtest::IntVecRefToBaseVectorProducer;
using edmtest::IntVecPtrVectorProducer;
using edmtest::ProdigalProducer;
DEFINE_FWK_MODULE(FailingProducer);
DEFINE_FWK_MODULE(NonAnalyzer);
DEFINE_FWK_MODULE(NonProducer);
DEFINE_FWK_MODULE(IntProducer);
DEFINE_FWK_MODULE(EventNumberIntProducer);
DEFINE_FWK_MODULE(TransientIntProducer);
DEFINE_FWK_MODULE(IntProducerFromTransient);
DEFINE_FWK_MODULE(Int16_tProducer);
DEFINE_FWK_MODULE(ToyDoubleProducer);
DEFINE_FWK_MODULE(SCSimpleProducer);
DEFINE_FWK_MODULE(OVSimpleProducer);
DEFINE_FWK_MODULE(VSimpleProducer);
DEFINE_FWK_MODULE(AVSimpleProducer);
DEFINE_FWK_MODULE(DSVProducer);
DEFINE_FWK_MODULE(DSTVProducer);
DEFINE_FWK_MODULE(IntTestAnalyzer);
DEFINE_FWK_MODULE(SCSimpleAnalyzer);
DEFINE_FWK_MODULE(DSVAnalyzer);
DEFINE_FWK_MODULE(AddIntsProducer);
DEFINE_FWK_MODULE(IntVectorProducer);
DEFINE_FWK_MODULE(IntListProducer);
DEFINE_FWK_MODULE(IntDequeProducer);
DEFINE_FWK_MODULE(IntSetProducer);
DEFINE_FWK_MODULE(IntVecRefVectorProducer);
DEFINE_FWK_MODULE(IntVecRefToBaseVectorProducer);
DEFINE_FWK_MODULE(IntVecPtrVectorProducer);
DEFINE_FWK_MODULE(ProdigalProducer);

