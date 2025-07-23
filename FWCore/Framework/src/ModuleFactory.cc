// -*- C++ -*-
//
// Package:     Framework
//
// Author:      Chris Jones
// Created:     Wed May 25 19:27:44 EDT 2005

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"

namespace edm {
  namespace eventsetup {

    //
    // static member functions
    //
    std::string ModuleMakerTraits::name() { return "CMS EDM Framework ESModule"; }
    std::string const& ModuleMakerTraits::baseType() { return ParameterSetDescriptionFillerBase::kBaseForESProducer; }
    void ModuleMakerTraits::addTo(EventSetupProvider& iProvider,
                                  std::shared_ptr<ESProductResolverProvider> iComponent) {
      iProvider.add(iComponent);
    }
  }  // namespace eventsetup
}  // namespace edm
COMPONENTFACTORY_GET(edm::eventsetup::ModuleMakerTraits);
