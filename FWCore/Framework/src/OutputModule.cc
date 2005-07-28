/*----------------------------------------------------------------------
$Id: OutputModule.cc,v 1.3 2005/07/27 04:39:58 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/ProductRegistry.h"

namespace edm {
  OutputModule::OutputModule(ParameterSet const& pset, ProductRegistry const& reg) : preg_(&reg), descVec_(), groupSelector_(pset) {
    for (ProductRegistry::ProductList::const_iterator it = reg.productList().begin();
          it != reg.productList().end(); ++it) {
      if (selected(*it)) {
        descVec_.push_back(&*it);
      }
    }
  }

  OutputModule::~OutputModule() {
  }

}
