#include "FWCore/Framework/interface/EventForTransformer.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {

  EventForTransformer::EventForTransformer(EventPrincipal const& ep, ModuleCallingContext const* moduleCallingContext)
      : eventPrincipal_{ep}, mcc_{moduleCallingContext} {}

  BasicHandle EventForTransformer::get(edm::TypeID const& iTypeID, ProductResolverIndex iIndex) const {
    bool amb = false;
    return eventPrincipal_.getByToken(PRODUCT_TYPE, iTypeID, iIndex, false, amb, nullptr, mcc_);
  }

  void EventForTransformer::put(ProductResolverIndex index,
                                std::unique_ptr<WrapperBase> edp,
                                BasicHandle const& iGetHandle) {
    eventPrincipal_.put(index, std::move(edp), iGetHandle.provenance()->productProvenance()->parentageID());
  }

}  // namespace edm
