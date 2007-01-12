#include <cassert>

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

  void ViewAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&) 
  {
    typedef SCSimpleProduct        sequence_t;
    typedef sequence_t::value_type value_type;
    typedef View<value_type>       view_t;

    string input("simple");
    Handle<sequence_t> hprod;
    e.getByLabel(input, hprod);
    assert(hprod.isValid());

    assert(e.size() == 1);

    Handle<view_t> hview;
    e.getByLabel(input, hview);
    assert(hview.isValid());

    assert(hprod.id() == hview.id());
    assert(*hprod.provenance() == *hview.provenance());

    assert(hprod->size() == hview->size());

    for (sequence_t::size_type i = 0;
	 i < hprod->size();
	 ++i)
      {

	value_type const& prod = (*hprod.product())[i];
	value_type const& view = (*hview.product())[i];
	assert(prod == view);
      }

  }
}

using edmtest::ViewAnalyzer;
DEFINE_FWK_MODULE(ViewAnalyzer);
