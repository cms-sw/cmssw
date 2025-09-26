// -*- C++ -*-
//
// Package:     Framework
//
// Author:      Chris Jones
// Created:     Wed May 25 19:27:37 EDT 2005
//

// user include files
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"

namespace edm {
  namespace eventsetup {

    //
    // static member functions
    //
    std::string SourceMakerTraits::name() { return "CMS EDM Framework ESSource"; }
    std::string const& SourceMakerTraits::baseType() { return ParameterSetDescriptionFillerBase::kBaseForESSource; }

  }  // namespace eventsetup
}  // namespace edm

COMPONENTFACTORY_GET(edm::eventsetup::SourceMakerTraits);
