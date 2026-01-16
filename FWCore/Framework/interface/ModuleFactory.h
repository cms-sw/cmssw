// -*- C++ -*-
#ifndef FWCore_Framework_ModuleFactory_h
#define FWCore_Framework_ModuleFactory_h
//
// Package:     Framework
//
/*
 Description:
    A ModuleFactory is a ComponentFactory used to construct
    modules that are ESProducers used by the EventSetup
    system.

    The addTo function will both construct a module and
    then pass a shared pointer to it to the EventSetupProvider.
    The ModuleFactory uses a Maker to accomplish this.
    There is one Maker associated with each type of
    ESProducer. The ComponentFactory stores the Makers.
    When the ComponentFactory needs a Maker it does not
    already have, it uses the plugin system to create it.

 Usage:
    addTo is called during EventProcessor construction
    for each configured ESProducer. The call stack looks
    similar to this:
        ...
        EventSetupsController::makeProvider
        fillEventSetupProvider (a free function)
        ComponentFactory::addTo
        ComponentMaker::addTo
        ModuleMakerTraits::addTo
*/
//
// Author:      Chris Jones
// Created:     Wed May 25 18:01:31 EDT 2005
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/ComponentFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"

// forward declarations
namespace edm {

  namespace eventsetup {
    class ESProductResolverProvider;
    class EventSetupProvider;

    struct ModuleMakerTraits {
      typedef ESProductResolverProvider base_type;

      static std::string name();
      static std::string const& baseType();
      static void addTo(EventSetupProvider& iProvider, std::shared_ptr<ESProductResolverProvider> iComponent);
    };
    template <class TType>
    struct ModuleMaker : public ComponentMaker<edm::eventsetup::ModuleMakerTraits, TType> {};

    typedef ComponentFactory<ModuleMakerTraits> ModuleFactory;
    typedef edmplugin::PluginFactory<edm::eventsetup::ComponentMakerBase<ModuleMakerTraits>*()> ModulePluginFactory;
  }  // namespace eventsetup
}  // namespace edm

#define DEFINE_FWK_EVENTSETUP_MODULE(type)                                                            \
  DEFINE_EDM_PLUGIN(edm::eventsetup::ModulePluginFactory, edm::eventsetup::ModuleMaker<type>, #type); \
  DEFINE_DESC_FILLER_FOR_ESPRODUCERS(type)

#endif
