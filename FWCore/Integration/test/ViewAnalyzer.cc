#include <algorithm>
#include <cassert>
#include <vector>
#include <list>
#include <deque>
#include <set>

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Integration/test/ViewAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/View.h"
#include "FWCore/Framework/interface/MakerMacros.h"

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
}

using edmtest::ViewAnalyzer;
DEFINE_FWK_MODULE(ViewAnalyzer);
