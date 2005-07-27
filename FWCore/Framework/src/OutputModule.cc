/*----------------------------------------------------------------------
$Id: OutputModule.cc,v 1.2 2005/07/14 22:50:53 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/ProductRegistry.h"

namespace edm {
  OutputModule::~OutputModule() {
  }

  void
  OutputModule::setProductRegistry(ProductRegistry & reg_) {
    reg_.sort();
    preg_ = &reg_;  
    for (ProductRegistry::ProductList::const_iterator it = reg_.productList().begin();
          it != reg_.productList().end(); ++it) {
      if (selected(*it)) {
        descVec_.push_back(&*it);
      }
    }
  }
}
