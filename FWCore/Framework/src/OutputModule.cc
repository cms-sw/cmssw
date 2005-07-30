/*----------------------------------------------------------------------
$Id: OutputModule.cc,v 1.4 2005/07/28 19:35:38 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/ProductRegistry.h"

namespace edm {
  OutputModule::OutputModule(ParameterSet const& pset, ProductRegistry const& reg) : preg_(&reg), descVec_(), groupSelector_(pset) {
    for (ProductRegistry::ProductList::const_iterator it = reg.productList().begin();
          it != reg.productList().end(); ++it) {
      if (selected(it->second)) {
        descVec_.push_back(&it->second);
      }
    }
  }

  OutputModule::~OutputModule() {
  }

}
