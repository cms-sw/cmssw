#ifndef FWCore_Framework_ModuleFactory_h
#define FWCore_Framework_ModuleFactory_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ModuleFactory
//
/**\class ModuleFactory ModuleFactory.h FWCore/Framework/interface/ModuleFactory.h

 Description: Factory which is dynamically loadable and used to create an eventstore module

 Usage:
    Used by the EDM plugin-manager

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
  class ParameterSet;

  namespace eventsetup {
    class DataProxyProvider;
    class EventSetupsController;

    struct ModuleMakerTraits {
      typedef DataProxyProvider base_type;

      static std::string name();
      static std::string const& baseType();
      static void addTo(EventSetupProvider& iProvider,
                        std::shared_ptr<DataProxyProvider> iComponent,
                        ParameterSet const&,
                        bool);
      static void replaceExisting(EventSetupProvider& iProvider, std::shared_ptr<DataProxyProvider> iComponent);
      static std::shared_ptr<base_type> getComponentAndRegisterProcess(EventSetupsController& esController,
                                                                       ParameterSet const& iConfiguration);
      static void putComponent(EventSetupsController& esController,
                               ParameterSet& iConfiguration,
                               std::shared_ptr<base_type> const& component);
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
