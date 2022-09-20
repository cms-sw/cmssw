#ifndef FWCore_Framework_EventForTransformer_h
#define FWCore_Framework_EventForTransformer_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventForTransformer
//
/**\class edm::EventForTransformer

*/
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/WrapperBase.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"

#include <memory>

namespace edm {

  class EventPrincipal;
  class ModuleCallingContext;

  class EventForTransformer {
  public:
    EventForTransformer(EventPrincipal const&, ModuleCallingContext const*);

    BasicHandle get(edm::TypeID const& iTypeID, ProductResolverIndex iIndex) const;

    void put(ProductResolverIndex index, std::unique_ptr<WrapperBase> edp, BasicHandle const& iGetHandle);

  private:
    EventPrincipal const& eventPrincipal_;
    ModuleCallingContext const* mcc_;
  };
}  // namespace edm
#endif
