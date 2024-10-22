// -*- C++ -*-
//
// Package:     Framework
// Class  :     LooperFactory
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Wed May 25 19:27:37 EDT 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"

//
// static member functions
//
namespace edm {
  namespace eventsetup {
    std::string LooperMakerTraits::name() { return "CMS EDM Framework EDLooper"; }
    std::string const& LooperMakerTraits::baseType() { return ParameterSetDescriptionFillerBase::kBaseForEDLooper; }

    void LooperMakerTraits::replaceExisting(EventSetupProvider&, std::shared_ptr<EDLooperBase>) {
      throw edm::Exception(edm::errors::LogicError) << "LooperMakerTraits::replaceExisting\n"
                                                    << "This function is not implemented and should never be called.\n"
                                                    << "Please report this to a Framework Developer\n";
    }

    std::shared_ptr<LooperMakerTraits::base_type> LooperMakerTraits::getComponentAndRegisterProcess(
        EventSetupsController&, ParameterSet const&) {
      return std::shared_ptr<LooperMakerTraits::base_type>();
    }

    void LooperMakerTraits::putComponent(EventSetupsController&,
                                         ParameterSet const&,
                                         std::shared_ptr<base_type> const&) {}
  }  // namespace eventsetup
}  // namespace edm

COMPONENTFACTORY_GET(edm::eventsetup::LooperMakerTraits);
