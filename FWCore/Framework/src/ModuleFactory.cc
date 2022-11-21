// -*- C++ -*-
//
// Package:     Framework
// Class  :     ModuleFactory
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Wed May 25 19:27:44 EDT 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupsController.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"

//
// constants, enums and typedefs
//
namespace edm {
  namespace eventsetup {

    //
    // static member functions
    //
    std::string ModuleMakerTraits::name() { return "CMS EDM Framework ESModule"; }
    std::string const& ModuleMakerTraits::baseType() { return ParameterSetDescriptionFillerBase::kBaseForESProducer; }
    void ModuleMakerTraits::addTo(EventSetupProvider& iProvider,
                                  std::shared_ptr<DataProxyProvider> iComponent,
                                  ParameterSet const&,
                                  bool) {
      iProvider.add(iComponent);
    }

    void ModuleMakerTraits::replaceExisting(EventSetupProvider& iProvider,
                                            std::shared_ptr<DataProxyProvider> iComponent) {
      iProvider.replaceExisting(iComponent);
    }

    std::shared_ptr<ModuleMakerTraits::base_type> ModuleMakerTraits::getComponentAndRegisterProcess(
        EventSetupsController& esController, ParameterSet const& iConfiguration) {
      return esController.getESProducerAndRegisterProcess(iConfiguration, esController.indexOfNextProcess());
    }

    void ModuleMakerTraits::putComponent(EventSetupsController& esController,
                                         ParameterSet& iConfiguration,
                                         std::shared_ptr<base_type> const& component) {
      esController.putESProducer(iConfiguration, component, esController.indexOfNextProcess());
    }
  }  // namespace eventsetup
}  // namespace edm
COMPONENTFACTORY_GET(edm::eventsetup::ModuleMakerTraits);
