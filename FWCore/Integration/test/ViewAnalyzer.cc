#include <algorithm>
#include <cassert>
#include <vector>
#include <list>
#include <deque>
#include <set>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/View.h"
#include "FWCore/Integration/test/ViewAnalyzer.h"

using namespace edm;
using namespace std;


namespace edmtest 
{

  ViewAnalyzer::ViewAnalyzer(edm::ParameterSet const&) 
  { }

  ViewAnalyzer::~ViewAnalyzer() 
  { }

  void 
  ViewAnalyzer::analyze(edm::Event const& e, 
			edm::EventSetup const& /* unused */)
  {
    assert(e.size() > 0);

    testProduct<SCSimpleProduct>(e, "simple");
    testProduct<OVSimpleProduct>(e, "ovsimple");
    testProduct<AVSimpleProduct>(e, "avsimple");
    testProduct<std::vector<int> >(e, "intvec");
    testProduct<std::list<int> >(e, "intlist");
    testProduct<std::deque<int> >(e, "intdeque");
    testProduct<std::set<int> >(e, "intset");

    testDSVProduct(e, "dsvsimple");

    testProductWithBaseClass(e, "ovsimple");

    //testProduct<edm::RefVector<std::vector<int> > >(e, "intvecrefvec");
  }

  template <class P>
  void
  ViewAnalyzer::testProduct(edm::Event const& e,
 			    std::string const& moduleLabel) const
  {
    typedef P                               sequence_t;
    typedef typename sequence_t::value_type value_t;
    typedef View<value_t>                   view_t;
    
    Handle<sequence_t> hprod;
    e.getByLabel(moduleLabel, hprod);
    assert(hprod.isValid());
    
    Handle<view_t> hview;
    e.getByLabel(moduleLabel, hview);
    assert(hview.isValid());
    
    assert(hprod.id() == hview.id());
    assert(*hprod.provenance() == *hview.provenance());
    
    assert(hprod->size() == hview->size());

    typename sequence_t::const_iterator i_prod = hprod->begin();
    typename sequence_t::const_iterator e_prod = hprod->end();
    typename view_t::const_iterator     i_view = hview->begin();
    typename view_t::const_iterator     e_view = hview->end();

    while ( i_prod != e_prod && i_view != e_view)
      {
	value_t const& prod = *i_prod;
	value_t const& view = *i_view;
        assert(prod == view);

	++i_prod; ++i_view;
      }
  }

  void
  ViewAnalyzer::testDSVProduct(edm::Event const& e,
 			    std::string const& moduleLabel) const
  {
    typedef edmtest::DSVSimpleProduct sequence_t;
    typedef sequence_t::value_type    value_t;
    typedef View<value_t>             view_t;
    
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
    view_t::const_iterator     i_view = hview->begin();
    view_t::const_iterator     e_view = hview->end();

    while ( i_prod != e_prod && i_view != e_view)
      {
	value_t const& prod = *i_prod;
	value_t const& view = *i_view;
        assert(prod.detId() == view.detId());
        assert(prod.data == view.data);

	++i_prod; ++i_view;
      }
  }

  // The point of this one is to test that a one can get
  // a View of "Simple" objects even when the sequence
  // has elements of a different type. The different type
  // inherits from "Simple" and is named "SimpleDerived"
  void
  ViewAnalyzer::testProductWithBaseClass(edm::Event const& e,
 			    std::string const& moduleLabel) const
  {
    typedef OVSimpleDerivedProduct          sequence_t;
    typedef Simple                          value_t;
    typedef View<value_t>                   view_t;
    
    Handle<sequence_t> hprod;
    e.getByLabel(moduleLabel, "derived", hprod);
    assert(hprod.isValid());
    
    Handle<view_t> hview;
    e.getByLabel(moduleLabel, "derived", hview);
    assert(hview.isValid());
    
    assert(hprod.id() == hview.id());
    assert(*hprod.provenance() == *hview.provenance());
    
    assert(hprod->size() == hview->size());

    sequence_t::const_iterator i_prod = hprod->begin();
    sequence_t::const_iterator e_prod = hprod->end();
    view_t::const_iterator     i_view = hview->begin();
    view_t::const_iterator     e_view = hview->end();

    while ( i_prod != e_prod && i_view != e_view)
      {
	SimpleDerived const& prod = *i_prod;
	Simple const& view = *i_view;
        assert(prod == view);

	++i_prod; ++i_view;
      }
  }
}

using edmtest::ViewAnalyzer;
DEFINE_FWK_MODULE(ViewAnalyzer);
