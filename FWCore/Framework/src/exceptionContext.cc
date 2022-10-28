#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ComponentDescription.h"

#include <ostream>

namespace edm {

  void exceptionContext(cms::Exception& ex, ESModuleCallingContext const& mcc) {
    ESModuleCallingContext const* imcc = &mcc;
    while (true) {
      std::ostringstream iost;
      if (imcc->state() == ESModuleCallingContext::State::kPrefetching) {
        iost << "Prefetching for EventSetup module ";
      } else {
        iost << "Calling method for EventSetup module ";
      }
      iost << imcc->componentDescription()->type_ << "/'" << imcc->componentDescription()->label_ << "'";

      ex.addContext(iost.str());
      if (imcc->type() != ESParentContext::Type::kESModule) {
        break;
      }
      imcc = imcc->esmoduleCallingContext();
    }
    edm::exceptionContext(ex, *imcc->moduleCallingContext());
  }
}  // namespace edm
