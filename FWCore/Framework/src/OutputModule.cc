/*----------------------------------------------------------------------
$Id: OutputModule.cc,v 1.6 2005/08/25 20:26:19 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

namespace edm {
  OutputModule::OutputModule(ParameterSet const& pset) : nextID_(), descVec_(), groupSelector_(pset) {
    Service<ConstProductRegistry> reg;
    nextID_ = reg->nextID();
     
    for (ProductRegistry::ProductList::const_iterator it = reg->productList().begin();
          it != reg->productList().end(); ++it) {
      if (selected(it->second)) {
        descVec_.push_back(&it->second);
      }
    }
  }

  OutputModule::~OutputModule() {
  }

  void OutputModule::beginJob(EventSetup const&) {
  }

  void OutputModule::endJob() {
  }
}
