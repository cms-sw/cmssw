// -*- C++ -*-
#ifndef FWCore_Framework_SourceFactory_h
#define FWCore_Framework_SourceFactory_h
//
// Package:     Framework
//
/*
 Description:
    A SourceFactory is a ComponentFactory used to construct
    modules that are ESSources used by the EventSetup
    system.

    The addTo function will both construct anESSource and
    then pass a shared pointer to it to the EventSetupProvider.
    The SourceFactory uses a Maker to accomplish this.
    There is one Maker associated with each type of
    ESSource. The ComponentFactory stores the Makers.
    When the ComponentFactory needs a Maker it does not
    already have, it uses the plugin system to create it.

 Usage:
    addTo is called during EventProcessor construction
    for each configured ESSource. The call stack looks
    similar to this:
        ...
        EventSetupsController::makeProvider
        fillEventSetupProvider (a free function)
        ComponentFactory::addTo
        ComponentMaker::addTo
        SourceMakerTraits::addTo
*/
//
// Author:      Chris Jones
// Created:     Wed May 25 18:01:38 EDT 2005
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/ComponentFactory.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"

// forward declarations

namespace edm {
  class EventSetupRecordIntervalFinder;

  namespace eventsetup {
    class ESProductResolverProvider;

    template <class T>
    void addProviderTo(EventSetupProvider& iProvider, std::shared_ptr<T> iComponent, const ESProductResolverProvider*) {
      std::shared_ptr<ESProductResolverProvider> pProvider(iComponent);
      ComponentDescription description = pProvider->description();
      description.isSource_ = true;
      pProvider->setDescription(description);
      iProvider.add(pProvider);
    }
    template <class T>
    void addProviderTo(EventSetupProvider& /* iProvider */, std::shared_ptr<T> /*iComponent*/, const void*) {
      //do nothing
    }

    struct SourceMakerTraits {
      typedef EventSetupRecordIntervalFinder base_type;
      static std::string name();
      static std::string const& baseType();
      template <class T>
      static void addTo(EventSetupProvider& iProvider, std::shared_ptr<T> iComponent) {
        //a source does not always have to be a provider
        addProviderTo(iProvider, iComponent, static_cast<const T*>(nullptr));
        std::shared_ptr<EventSetupRecordIntervalFinder> pFinder(iComponent);
        iProvider.add(pFinder);
      }
    };

    template <class TType>
    struct SourceMaker : public ComponentMaker<edm::eventsetup::SourceMakerTraits, TType> {};
    typedef ComponentFactory<SourceMakerTraits> SourceFactory;

    typedef edmplugin::PluginFactory<edm::eventsetup::ComponentMakerBase<edm::eventsetup::SourceMakerTraits>*()>
        SourcePluginFactory;
  }  // namespace eventsetup
}  // namespace edm

#define DEFINE_FWK_EVENTSETUP_SOURCE(type)                                                            \
  DEFINE_EDM_PLUGIN(edm::eventsetup::SourcePluginFactory, edm::eventsetup::SourceMaker<type>, #type); \
  DEFINE_DESC_FILLER_FOR_ESSOURCES(type)

#endif
