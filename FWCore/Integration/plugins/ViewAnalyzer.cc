#include <algorithm>
#include <cassert>
#include <deque>
#include <list>
#include <set>
#include <typeinfo>
#include <utility>
#include <vector>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;
using namespace std::rel_ops;

namespace edmtest {
  class ViewAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit ViewAnalyzer(edm::ParameterSet const& /* no parameters*/);
    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const& /* unused */) const override;

    template <typename P, typename V>
    void testProduct(edm::Event const& e, std::string const& moduleLabel) const;

    void testDSVProduct(edm::Event const& e, std::string const& moduleLabel) const;

    void testAVProduct(edm::Event const& e, std::string const& moduleLabel) const;

    void testProductWithBaseClass(edm::Event const& e, std::string const& moduleLabel) const;

    void testRefVector(edm::Event const& e, std::string const& moduleLabel) const;

    void testRefToBaseVector(edm::Event const& e, std::string const& moduleLabel) const;

    void testPtrVector(edm::Event const& e, std::string const& moduleLabel) const;

    void testStdVectorPtr(edm::Event const& e, std::string const& moduleLabel) const;

    void testStdVectorUniquePtr(edm::Event const& e, std::string const& moduleLabel) const;
  };

  ViewAnalyzer::ViewAnalyzer(ParameterSet const&) {
    consumes<edm::View<int>>(edm::InputTag{"intvec", "", "TEST"});
    consumes<edm::View<int>>(edm::InputTag{"intvec", ""});
    consumes<std::vector<int>>(edm::InputTag{"intvec"});
    consumes<edm::View<int>>(edm::InputTag{"intvec"});
    consumes<std::list<int>>(edm::InputTag{"intlist"});
    consumes<edm::View<int>>(edm::InputTag{"intlist"});
    consumes<std::deque<int>>(edm::InputTag{"intdeque"});
    consumes<edm::View<int>>(edm::InputTag{"intdeque"});
    consumes<std::set<int>>(edm::InputTag{"intset"});
    consumes<edm::View<int>>(edm::InputTag{"intset"});
    consumes<SCSimpleProduct>(edm::InputTag{"simple"});
    consumes<edm::View<SCSimpleProduct::value_type>>(edm::InputTag{"simple"});
    consumes<OVSimpleProduct>(edm::InputTag{"ovsimple"});
    consumes<edm::View<OVSimpleProduct::value_type>>(edm::InputTag{"ovsimple"});
    consumes<AVSimpleProduct>(edm::InputTag{"avsimple"});
    consumes<edm::View<AVSimpleProduct::value_type>>(edm::InputTag{"avsimple"});
    consumes<edmtest::DSVSimpleProduct>(edm::InputTag{"dsvsimple"});
    consumes<edm::View<edmtest::DSVSimpleProduct::value_type>>(edm::InputTag{"dsvsimple"});

    consumes<OVSimpleDerivedProduct>(edm::InputTag{"ovsimple", "derived"});
    consumes<edm::View<Simple>>(edm::InputTag{"ovsimple", "derived"});

    consumes<RefVector<std::vector<int>>>(edm::InputTag{"intvecrefvec"});
    consumes<edm::View<int>>(edm::InputTag{"intvecrefvec"});

    consumes<RefToBaseVector<int>>(edm::InputTag{"intvecreftbvec"});
    consumes<edm::View<int>>(edm::InputTag{"intvecreftbvec"});

    consumes<PtrVector<int>>(edm::InputTag{"intvecptrvec"});
    consumes<edm::View<int>>(edm::InputTag{"intvecptrvec"});

    consumes<std::vector<edm::Ptr<int>>>(edm::InputTag{"intvecstdvecptr"});
    consumes<edm::View<int>>(edm::InputTag{"intvecstdvecptr"});

    consumes<std::vector<std::unique_ptr<int>>>(edm::InputTag{"intvecstdvecuniqptr"});
    consumes<edm::View<int>>(edm::InputTag{"intvecstdvecuniqptr"});
    consumes<std::vector<std::unique_ptr<IntProduct>>>(edm::InputTag{"intvecstdvecuniqptr"});
    consumes<edm::View<IntProduct>>(edm::InputTag{"intvecstdvecuniqptr"});

    mayConsume<edm::View<int>>(edm::InputTag{"intvecptrvecdoesNotExist"});
  }

  template <typename P, typename V = typename P::value_type>
  struct tester {
    static void call(ViewAnalyzer const* va, Event const& e, char const* moduleLabel) {
      va->template testProduct<P, V>(e, moduleLabel);
    }
  };

  void ViewAnalyzer::analyze(StreamID, Event const& e, EventSetup const& /* unused */) const {
    tester<std::vector<int>>::call(this, e, "intvec");
    tester<std::list<int>>::call(this, e, "intlist");
    tester<std::deque<int>>::call(this, e, "intdeque");
    tester<std::set<int>>::call(this, e, "intset");

    tester<SCSimpleProduct>::call(this, e, "simple");
    tester<OVSimpleProduct>::call(this, e, "ovsimple");
    tester<AVSimpleProduct>::call(this, e, "avsimple");

    testDSVProduct(e, "dsvsimple");
    testAVProduct(e, "avsimple");
    testProductWithBaseClass(e, "ovsimple");
    testRefVector(e, "intvecrefvec");
    testRefToBaseVector(e, "intvecreftbvec");
    testPtrVector(e, "intvecptrvec");
    testStdVectorPtr(e, "intvecstdvecptr");
    testStdVectorUniquePtr(e, "intvecstdvecuniqptr");

    //See if InputTag works
    {
      edm::InputTag tag("intvec", "");
      edm::Handle<edm::View<int>> hInt;
      e.getByLabel(tag, hInt);
      assert(hInt.isValid());
    }
    {
      edm::InputTag tag("intvec", "", "TEST");
      edm::Handle<edm::View<int>> hInt;
      e.getByLabel(tag, hInt);
      assert(hInt.isValid());
    }
  }

  template <typename P, typename V>
  void ViewAnalyzer::testProduct(Event const& e, std::string const& moduleLabel) const {
    typedef P sequence_t;
    typedef V value_t;
    typedef View<value_t> view_t;

    Handle<sequence_t> hproduct;
    e.getByLabel(moduleLabel, hproduct);
    assert(hproduct.isValid());

    Handle<view_t> hview;
    e.getByLabel(moduleLabel, hview);
    assert(hview.isValid());

    assert(hproduct.id() == hview.id());
    assert(*hproduct.provenance() == *hview.provenance());

    assert(hproduct->size() == hview->size());

    typename sequence_t::const_iterator i_product = hproduct->begin();
    typename sequence_t::const_iterator e_product = hproduct->end();
    typename view_t::const_iterator i_view = hview->begin();
    typename view_t::const_iterator e_view = hview->end();
    size_t slot = 0;
    while (i_product != e_product && i_view != e_view) {
      value_t const& product_item = *i_product;
      value_t const& view_item = *i_view;
      assert(product_item == view_item);

      edm::Ref<sequence_t> ref3(hproduct, slot);
      assert(*ref3 == product_item);

      edm::RefProd<sequence_t> refProd4(hproduct);
      edm::Ref<sequence_t> ref4(refProd4, slot);
      assert(*ref4 == product_item);

      ++i_product;
      ++i_view;
      ++slot;
    }

    // Make sure the references are right.
    size_t numElements = hview->size();
    for (size_t i = 0; i < numElements; ++i) {
      RefToBase<value_t> ref = hview->refAt(i);
      assert(ref.isNonnull());
    }
  }

  void ViewAnalyzer::testDSVProduct(Event const& e, std::string const& moduleLabel) const {
    typedef edmtest::DSVSimpleProduct sequence_t;
    typedef sequence_t::value_type value_t;
    typedef View<value_t> view_t;

    Handle<sequence_t> hprod;
    e.getByLabel(moduleLabel, hprod);
    assert(hprod.isValid());

    Handle<view_t> hview;
    e.getByLabel(moduleLabel, hview);
    assert(hview.isValid());

    assert(hprod.id() == hview.id());
    assert(*hprod.provenance() == *hview.provenance());

    assert(hprod->size() == hview->size());

    sequence_t::const_iterator i_prod = hprod->begin();
    sequence_t::const_iterator e_prod = hprod->end();
    view_t::const_iterator i_view = hview->begin();
    view_t::const_iterator e_view = hview->end();

    while (i_prod != e_prod && i_view != e_view) {
      value_t const& prod = *i_prod;
      value_t const& view = *i_view;
      assert(prod.detId() == view.detId());
      assert(prod.data == view.data);

      ++i_prod;
      ++i_view;
    }
  }

  void ViewAnalyzer::testAVProduct(Event const& e, std::string const& moduleLabel) const {
    typedef edmtest::AVSimpleProduct sequence_t;
    typedef sequence_t::value_type value_t;
    typedef View<value_t> view_t;

    Handle<sequence_t> hprod;
    e.getByLabel(moduleLabel, hprod);
    assert(hprod.isValid());

    Handle<view_t> hview;
    e.getByLabel(moduleLabel, hview);
    assert(hview.isValid());

    assert(hprod.id() == hview.id());
    assert(*hprod.provenance() == *hview.provenance());

    assert(hprod->size() == hview->size());

    sequence_t::const_iterator i_prod = hprod->begin();
    sequence_t::const_iterator e_prod = hprod->end();
    view_t::const_iterator i_view = hview->begin();
    view_t::const_iterator e_view = hview->end();

    while (i_prod != e_prod && i_view != e_view) {
      value_t const& prod = *i_prod;
      value_t const& view = *i_view;
      assert(prod == view);
      assert((*hprod)[prod.first] == prod.second);
      edm::Ptr<sequence_t::key_type> ptr(prod.first.id(), prod.first.key(), &e.productGetter());
      assert((*hprod)[ptr] == prod.second);
      edm::RefToBase<sequence_t::key_type> refToBase(prod.first);
      assert((*hprod)[refToBase] == prod.second);
      ++i_prod;
      ++i_view;
    }
  }

  // The point of this one is to test that a one can get
  // a View of "Simple" objects even when the sequence
  // has elements of a different type. The different type
  // inherits from "Simple" and is named "SimpleDerived"
  void ViewAnalyzer::testProductWithBaseClass(Event const& e, std::string const& moduleLabel) const {
    typedef OVSimpleDerivedProduct sequence_t;
    typedef Simple value_t;
    typedef View<value_t> view_t;

    Handle<sequence_t> hprod;
    e.getByLabel(moduleLabel, "derived", hprod);
    assert(hprod.isValid());

    Handle<view_t> hview;
    e.getByLabel(moduleLabel, "derived", hview);
    assert(hview.isValid());

    assert(hprod.id() == hview.id());
    assert(*hprod.provenance() == *hview.provenance());

    assert(hprod->size() == hview->size());

    unsigned slot = 0;

    sequence_t::const_iterator i_prod = hprod->begin();
    sequence_t::const_iterator e_prod = hprod->end();
    view_t::const_iterator i_view = hview->begin();
    view_t::const_iterator e_view = hview->end();

    while (i_prod != e_prod && i_view != e_view) {
      SimpleDerived const& prod = *i_prod;
      Simple const& view = *i_view;
      assert(prod == view);

      // Tack on a test of RefToBase::castTo here

      edm::RefToBaseProd<Simple> refToBaseProd(hview);
      edm::RefToBase<Simple> refToBase(refToBaseProd, slot);

      edm::Ptr<SimpleDerived> ptr = refToBase.castTo<edm::Ptr<SimpleDerived>>();
      SimpleDerived const& valueFromPtr = *ptr;
      assert(valueFromPtr == view);

      edm::Ref<edm::OwnVector<SimpleDerived>> ref = refToBase.castTo<edm::Ref<edm::OwnVector<SimpleDerived>>>();
      SimpleDerived const& valueFromRef = *ref;
      assert(valueFromRef == view);

      ++i_prod;
      ++i_view;
      ++slot;
    }
  }

  void ViewAnalyzer::testRefVector(Event const& e, std::string const& moduleLabel) const {
    typedef RefVector<std::vector<int>> sequence_t;
    typedef int value_t;
    typedef View<value_t> view_t;

    Handle<sequence_t> hproduct;
    e.getByLabel(moduleLabel, hproduct);
    assert(hproduct.isValid());

    Handle<view_t> hview;
    e.getByLabel(moduleLabel, hview);
    assert(hview.isValid());

    assert(hproduct.id() == hview.id());
    assert(*hproduct.provenance() == *hview.provenance());

    assert(hproduct->size() == hview->size());

    sequence_t::const_iterator i_product = hproduct->begin();
    sequence_t::const_iterator e_product = hproduct->end();
    view_t::const_iterator i_view = hview->begin();
    view_t::const_iterator e_view = hview->end();
    size_t slot = 0;
    while (i_product != e_product && i_view != e_view) {
      value_t const& product_item = **i_product;
      value_t const& view_item = *i_view;
      assert(product_item == view_item);

      // Tack on a test of RefToBase::castTo here
      edm::RefToBaseProd<int> refToBaseProd(hview);
      edm::RefToBase<int> refToBase(refToBaseProd, slot);

      edm::Ptr<int> ref = refToBase.castTo<edm::Ptr<int>>();
      int item_other = *ref;
      assert(item_other == product_item);

      edm::Ref<std::vector<int>> ref2 = refToBase.castTo<edm::Ref<std::vector<int>>>();
      int item_other2 = *ref2;
      assert(item_other2 == product_item);

      edm::Ref<sequence_t> ref3(hproduct, slot);
      assert(*ref3 == product_item);

      ++i_product;
      ++i_view;
      ++slot;
    }
  }

  void ViewAnalyzer::testRefToBaseVector(Event const& e, std::string const& moduleLabel) const {
    typedef RefToBaseVector<int> sequence_t;
    typedef int value_t;
    typedef View<value_t> view_t;

    Handle<sequence_t> hproduct;
    e.getByLabel(moduleLabel, hproduct);
    assert(hproduct.isValid());

    Handle<view_t> hview;
    e.getByLabel(moduleLabel, hview);
    assert(hview.isValid());

    assert(hproduct.id() == hview.id());
    assert(*hproduct.provenance() == *hview.provenance());

    assert(hproduct->size() == hview->size());

    sequence_t::const_iterator i_product = hproduct->begin();
    sequence_t::const_iterator e_product = hproduct->end();
    view_t::const_iterator i_view = hview->begin();
    view_t::const_iterator e_view = hview->end();
    while (i_product != e_product && i_view != e_view) {
      value_t const& product_item = **i_product;
      value_t const& view_item = *i_view;
      assert(product_item == view_item);
      ++i_product;
      ++i_view;
    }
  }

  void ViewAnalyzer::testPtrVector(Event const& e, std::string const& moduleLabel) const {
    typedef PtrVector<int> sequence_t;
    typedef int value_t;
    typedef View<value_t> view_t;

    Handle<sequence_t> hproduct;
    e.getByLabel(moduleLabel, hproduct);
    assert(hproduct.isValid());

    Handle<view_t> hview;

    InputTag tag(moduleLabel + "doesNotExist");
    e.getByLabel(tag, hview);
    assert(!hview.isValid());

    e.getByLabel(moduleLabel + "doesNotExist", hview);
    assert(!hview.isValid());

    InputTag tag2(moduleLabel);
    e.getByLabel(tag2, hview);
    assert(hview.isValid());

    assert(hproduct.id() == hview.id());
    assert(*hproduct.provenance() == *hview.provenance());

    assert(hproduct->size() == hview->size());

    sequence_t::const_iterator i_product = hproduct->begin();
    sequence_t::const_iterator e_product = hproduct->end();
    view_t::const_iterator i_view = hview->begin();
    view_t::const_iterator e_view = hview->end();
    while (i_product != e_product && i_view != e_view) {
      value_t const& product_item = **i_product;
      value_t const& view_item = *i_view;
      assert(product_item == view_item);
      ++i_product;
      ++i_view;
    }
  }
}  // namespace edmtest

namespace {
  template <typename PtrT>
  struct ValueType {
    using type = typename PtrT::element_type;
  };

  template <typename T>
  struct ValueType<edm::Ptr<T>> {
    using type = T;
  };

  template <typename Ptr>
  void testStdVectorPtrT(Event const& e, std::string const& moduleLabel) {
    using sequence_t = std::vector<Ptr>;
    using value_t = typename ValueType<Ptr>::type;
    using view_t = View<value_t>;

    Handle<sequence_t> hproduct;
    e.getByLabel(moduleLabel, hproduct);
    assert(hproduct.isValid());

    Handle<view_t> hview;

    InputTag tag(moduleLabel);
    e.getByLabel(tag, hview);
    assert(hview.isValid());

    assert(hproduct.id() == hview.id());
    assert(*hproduct.provenance() == *hview.provenance());

    assert(hproduct->size() == hview->size());

    typename sequence_t::const_iterator i_product = hproduct->begin();
    typename sequence_t::const_iterator e_product = hproduct->end();
    typename view_t::const_iterator i_view = hview->begin();
    typename view_t::const_iterator e_view = hview->end();
    unsigned int slot = 0;
    while (i_product != e_product && i_view != e_view) {
      value_t const& product_item = **i_product;
      value_t const& view_item = *i_view;
      assert(product_item == view_item);

      edm::Ref<sequence_t> ref3(hproduct, slot);
      assert(**ref3 == product_item);

      ++i_product;
      ++i_view;
      ++slot;
    }
  }
}  // namespace

namespace edmtest {
  void ViewAnalyzer::testStdVectorPtr(Event const& e, std::string const& moduleLabel) const {
    testStdVectorPtrT<edm::Ptr<int>>(e, moduleLabel);
  }

  void ViewAnalyzer::testStdVectorUniquePtr(Event const& e, std::string const& moduleLabel) const {
    testStdVectorPtrT<std::unique_ptr<int>>(e, moduleLabel);
    testStdVectorPtrT<std::unique_ptr<IntProduct>>(e, moduleLabel);
  }

}  // namespace edmtest

using edmtest::ViewAnalyzer;
DEFINE_FWK_MODULE(ViewAnalyzer);
