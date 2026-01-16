// -*- C++ -*-
//
// Package:     Framework
//
// Author:      Chris Jones
// Created:     Wed May 25 19:27:37 EDT 2005
//

// user include files
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"

namespace edm {
  namespace eventsetup {

    //
    // static member functions
    //
    std::string LooperMakerTraits::name() { return "CMS EDM Framework EDLooper"; }
    std::string const& LooperMakerTraits::baseType() { return ParameterSetDescriptionFillerBase::kBaseForEDLooper; }

  }  // namespace eventsetup
}  // namespace edm

COMPONENTFACTORY_GET(edm::eventsetup::LooperMakerTraits);
