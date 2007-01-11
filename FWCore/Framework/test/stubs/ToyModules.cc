
/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest {

  //--------------------------------------------------------------------
  //
  // Toy producers
  //
  //--------------------------------------------------------------------

  //--------------------------------------------------------------------
  //
  class FailingProducer : public edm::EDProducer {
  public:
    explicit FailingProducer(edm::ParameterSet const& /*p*/) {  }
    virtual ~FailingProducer() { }
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
  };

  void
  FailingProducer::produce(edm::Event&, const edm::EventSetup&) {
    // We throw a double because the EventProcessor is *never*
    // configured to catch doubles.
    double x = 2.5;
    throw x;
  }

  //--------------------------------------------------------------------
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
    virtual ~IntProducer() { }
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
  private:
    int value_;
  };

  void
  IntProducer::produce(edm::Event& e, const edm::EventSetup&) {
    // EventSetup is not used.
    std::auto_ptr<IntProduct> p(new IntProduct(value_));
    e.put(p);
  }
  
  //--------------------------------------------------------------------
  //
  class DoubleProducer : public edm::EDProducer {
  public:
    explicit DoubleProducer(edm::ParameterSet const& p) : 
      value_(p.getParameter<double>("dvalue")) {
      produces<DoubleProduct>();
    }
    explicit DoubleProducer(double d) : value_(d) {
      produces<DoubleProduct>();
    }
    virtual ~DoubleProducer() { }
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
  private:
    double value_;
  };

  void
  DoubleProducer::produce(edm::Event& e, const edm::EventSetup&) {
    // EventSetup is not used.
    // Get input
    edm::Handle<IntProduct> h;
    assert(!h.isValid());

    try {
      std::string emptyLabel;
      e.getByLabel(emptyLabel, h);
      assert ("Failed to throw necessary exception" == 0);
    }
    catch (edm::Exception& x) {
      assert(!h.isValid());
    }
    catch (...) {
      assert("Threw wrong exception" == 0);
    }

    // Make output
    std::auto_ptr<DoubleProduct> p(new DoubleProduct(value_));
    e.put(p);
  }

  //--------------------------------------------------------------------
  //
  class AddIntsProducer : public edm::EDProducer {
  public:
    explicit AddIntsProducer(edm::ParameterSet const& p) : 
      labels_(p.getParameter<std::vector<std::string> >("labels")) {
        produces<IntProduct>();
      }
    virtual ~AddIntsProducer() { }
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
  private:
    std::vector<std::string> labels_;
  };
  
  void
  AddIntsProducer::produce(edm::Event& e, const edm::EventSetup&) {
    // EventSetup is not used.
    int value =0;
    for(std::vector<std::string>::iterator itLabel=labels_.begin();
	itLabel!=labels_.end(); ++itLabel) {
      edm::Handle<IntProduct> anInt;
      e.getByLabel(*itLabel, anInt);
      value +=anInt->value;
    }
    std::auto_ptr<IntProduct> p(new IntProduct(value));
    e.put(p);
  }

  //--------------------------------------------------------------------
  //
  class SCSimpleProducer : public edm::EDProducer {
  public:
    explicit SCSimpleProducer(edm::ParameterSet const& p) : 
      size_(p.getParameter<int>("size")) 
    {
      produces<SCSimpleProduct>();
      assert ( size_ > 1 );
    }

    explicit SCSimpleProducer(int i) : size_(i) 
    {
      produces<SCSimpleProduct>();
      assert ( size_ > 1 );
    }

    virtual ~SCSimpleProducer() { }
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
    
  private:
    int size_;  // number of Simples to put in the collection
  };

  void
  SCSimpleProducer::produce(edm::Event& e, 
			    const edm::EventSetup& /* unused */) 
  {
    // Fill up a collection so that it is sorted *backwards*.
    std::vector<Simple> guts(size_);
    for (int i = 0; i < size_; ++i)
      {
	guts[i].key = size_ - i;
	guts[i].value = 1.5 * i;
      }

    // Verify that the vector is not sorted -- in fact, it is sorted
    // backwards!
    for (int i = 1; i < size_; ++i)
      {
	assert( guts[i-1].id() > guts[i].id());
      }

    std::auto_ptr<SCSimpleProduct> p(new SCSimpleProduct(guts));
    
    // Put the product into the Event, thus sorting it.
    e.put(p);

  }

  //--------------------------------------------------------------------
  //
  class DSVProducer : public edm::EDProducer 
  {
  public:

    explicit DSVProducer(edm::ParameterSet const& p) :
      size_(p.getParameter<int>("size"))
    {
      produces<DSVSimpleProduct>();
      produces<DSVWeirdProduct>();
      assert(size_ > 1);
    }
    
    explicit DSVProducer(int i) : size_(i)
    {
      produces<DSVSimpleProduct>();
      produces<DSVWeirdProduct>();
      assert(size_ > 1);
    }

    virtual ~DSVProducer() { }

    virtual void produce(edm::Event& e, edm::EventSetup const&);

  private:
    template <class PROD> void make_a_product(edm::Event& e);
    int size_;
  };

  void
  DSVProducer::produce(edm::Event& e, 
			     const edm::EventSetup& /* unused */)
  {
    this->make_a_product<DSVSimpleProduct>(e);
    this->make_a_product<DSVWeirdProduct>(e);
  }


  template <class PROD>
  void
  DSVProducer::make_a_product(edm::Event& e)
  {
    typedef PROD                     product_type;
    typedef typename product_type::value_type detset;
    typedef typename detset::value_type       value_type;

    // Fill up a collection so that it is sorted *backwards*.
    std::vector<value_type> guts(size_);
    for (int i = 0; i < size_; ++i)
      {
	guts[i].data = size_ - i;
      }

    // Verify that the vector is not sorted -- in fact, it is sorted
    // backwards!
    for (int i = 1; i < size_; ++i)
      {
	assert( guts[i-1].data > guts[i].data);
      }

    detset item(1); // this will get DetID 1
    item.data = guts;

    std::auto_ptr<product_type> p(new product_type());
    p->insert(item);
    
    // Put the product into the Event, thus sorting it ... or not,
    // depending upon the product type.
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
    IntTestAnalyzer(const edm::ParameterSet& iPSet) :
      value_(iPSet.getUntrackedParameter<int>("valueMustMatch")),
      moduleLabel_(iPSet.getUntrackedParameter<std::string>("moduleLabel")) {
      }
     
    void analyze(const edm::Event& iEvent, const edm::EventSetup&) {
      edm::Handle<IntProduct> handle;
      iEvent.getByLabel(moduleLabel_,handle);
      if(handle->value != value_) {
	throw cms::Exception("ValueMissMatch")
	  <<"The value for \""<<moduleLabel_<<"\" is "
	  <<handle->value <<" but it was supposed to be "<<value_;
      }
    }
  private:
    int value_;
    std::string moduleLabel_;
  };


  //--------------------------------------------------------------------
  //
  class SCSimpleAnalyzer : public edm::EDAnalyzer 
  {
  public:
    SCSimpleAnalyzer(const edm::ParameterSet& iPSet) { }

    virtual void
    analyze(edm::Event const& e, edm::EventSetup const&);
  };


  void
  SCSimpleAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&)
  {

    // Get the product back out; it should be sorted.
    edm::Handle<SCSimpleProduct> h;
    e.getByType(h);
    assert( h.isValid() );

    // Check the sorting. DO NOT DO THIS IN NORMAL CODE; we are
    // copying all the values out of the SortedCollection so we can
    // manipulate them via an interface different from
    // SortedCollection, just so that we can make sure the collection
    // is sorted.
    std::vector<Simple> after( h->begin(), h->end() );
    typedef std::vector<Simple>::size_type size_type;
    

    // Verify that the vector *is* sorted.
    
    for (size_type i = 1, end = after.size(); i < end; ++i)
      {
	assert( after[i-1].id() < after[i].id());
      }
  }

  //--------------------------------------------------------------------
  //
  class DSVAnalyzer : public edm::EDAnalyzer 
  {
  public:
    DSVAnalyzer(const edm::ParameterSet& iPSet) { }

    virtual void
    analyze(edm::Event const& e, edm::EventSetup const&);
  private:
    void do_sorted_stuff(edm::Event const& e);
    void do_unsorted_stuff(edm::Event const& e);
  };



  void
  DSVAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&)
  {
    do_sorted_stuff(e);
    do_unsorted_stuff(e);
  }

  void
  DSVAnalyzer::do_sorted_stuff(edm::Event const& e)
  {
    typedef DSVSimpleProduct         product_type;
    typedef product_type::value_type detset;
    typedef detset::value_type       value_type;
    // Get the product back out; it should be sorted.
    edm::Handle<product_type> h;
    e.getByType(h);
    assert( h.isValid() );

    // Check the sorting. DO NOT DO THIS IN NORMAL CODE; we are
    // copying all the values out of the DetSetVector's first DetSet so we can
    // manipulate them via an interface different from
    // DetSet, just so that we can make sure the collection
    // is sorted.
    std::vector<value_type> after( h->begin()->data.begin(),
				   h->begin()->data.end() );
    typedef std::vector<value_type>::size_type size_type;
    

    // Verify that the vector *is* sorted.
    
    for (size_type i = 1, end = after.size(); i < end; ++i)
      {
	assert( after[i-1].data < after[i].data);
      }
  }

  void
  DSVAnalyzer::do_unsorted_stuff(edm::Event const& e)
  {
    typedef DSVWeirdProduct         product_type;
    typedef product_type::value_type detset;
    typedef detset::value_type       value_type;
    // Get the product back out; it should be unsorted.
    edm::Handle<product_type> h;
    e.getByType(h);
    assert( h.isValid() );

    // Check the sorting. DO NOT DO THIS IN NORMAL CODE; we are
    // copying all the values out of the DetSetVector's first DetSet so we can
    // manipulate them via an interface different from
    // DetSet, just so that we can make sure the collection
    // is not sorted.
    std::vector<value_type> after( h->begin()->data.begin(),
				   h->begin()->data.end() );
    typedef std::vector<value_type>::size_type size_type;
    

    // Verify that the vector is reverse-sorted.
    
    for (size_type i = 1, end = after.size(); i < end; ++i)
      {
	assert( after[i-1].data > after[i].data);
      }
  }


}

using edmtest::FailingProducer;
using edmtest::IntProducer;
using edmtest::DoubleProducer;
using edmtest::SCSimpleProducer;
using edmtest::DSVProducer;
using edmtest::IntTestAnalyzer;
using edmtest::SCSimpleAnalyzer;
using edmtest::DSVAnalyzer;
using edmtest::AddIntsProducer;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(FailingProducer);
DEFINE_ANOTHER_FWK_MODULE(IntProducer);
DEFINE_ANOTHER_FWK_MODULE(DoubleProducer);
DEFINE_ANOTHER_FWK_MODULE(SCSimpleProducer);
DEFINE_ANOTHER_FWK_MODULE(DSVProducer);
DEFINE_ANOTHER_FWK_MODULE(IntTestAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(SCSimpleAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(DSVAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(AddIntsProducer);
