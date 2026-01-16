#include "FWCore/Framework/interface/maker/ModuleHolder.h"
#include "FWCore/Utilities/interface/Exception.h"

void edm::maker::ModuleHolder::registerThinnedAssociations(ProductRegistry const& registry,
                                                           ThinnedAssociationsHelper& helper) {
  try {
    implRegisterThinnedAssociations(registry, helper);
  } catch (cms::Exception& ex) {
    ex.addContext("Calling registerThinnedAssociations() for module " + moduleDescription().moduleLabel());
    throw ex;
  }
}
