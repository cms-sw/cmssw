// -*- C++ -*-
#ifndef FWCore_Framework_LooperFactory_h
#define FWCore_Framework_LooperFactory_h
//
// Package:     Framework
//
/*
 Description:
    A LooperFactory is a ComponentFactory used to construct
    a looper which might be used by the EventSetup system.

    The addTo function will both construct a looper and
    might then pass a shared pointer to it to the EventSetupProvider.
    The LooperFactory uses a Maker to accomplish this.
    There is one Maker associated with each type of
    looper. The ComponentFactory stores the Maker.
    When the ComponentFactory needs a Maker it does not
    already have, it uses the plugin system to create it.

 Usage:
    addTo is called during EventProcessor construction
    for the looper if one is configured. The call stack
    looks similar to this:
        ...
        fillLooper
        ComponentFactory::addTo
        ComponentMaker::addTo
        LooperMakerTraits::addTo
*/
//
// Author:      Chris Jones
// Created:     Wed May 25 18:01:38 EDT 2005
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/ComponentFactory.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"

// forward declarations
namespace edm {
  class EDLooperBase;
  class EventSetupRecordIntervalFinder;

  namespace eventsetup {

    class ESProductResolverProvider;

    namespace looper {
      template <class T>
      void addProviderTo(EventSetupProvider& iProvider,
                         std::shared_ptr<T> iComponent,
                         const ESProductResolverProvider*) {
        std::shared_ptr<ESProductResolverProvider> pProvider(iComponent);
        ComponentDescription description = pProvider->description();
        description.isSource_ = true;
        description.isLooper_ = true;
        if (description.label_ == "@main_looper") {
          //remove the 'hidden' label so that es_prefer statements will work
          description.label_ = "";
        }
        pProvider->setDescription(description);
        iProvider.add(pProvider);
      }
      template <class T>
      void addProviderTo(EventSetupProvider& /* iProvider */, std::shared_ptr<T> /*iComponent*/, const void*) {
        //do nothing
      }

      template <class T>
      void addFinderTo(EventSetupProvider& iProvider,
                       std::shared_ptr<T> iComponent,
                       const EventSetupRecordIntervalFinder*) {
        std::shared_ptr<EventSetupRecordIntervalFinder> pFinder(iComponent);

        ComponentDescription description = pFinder->descriptionForFinder();
        description.isSource_ = true;
        description.isLooper_ = true;
        if (description.label_ == "@main_looper") {
          //remove the 'hidden' label so that es_prefer statements will work
          description.label_ = "";
        }
        pFinder->setDescriptionForFinder(description);

        iProvider.add(pFinder);
      }
      template <class T>
      void addFinderTo(EventSetupProvider& /* iProvider */, std::shared_ptr<T> /*iComponent*/, const void*) {
        //do nothing
      }
    }  // namespace looper
    struct LooperMakerTraits {
      typedef EDLooperBase base_type;
      static std::string name();
      static std::string const& baseType();
      template <class T>
      static void addTo(EventSetupProvider& iProvider, std::shared_ptr<T> iComponent) {
        //a looper does not always have to be a provider or a finder
        looper::addProviderTo(iProvider, iComponent, static_cast<const T*>(nullptr));
        looper::addFinderTo(iProvider, iComponent, static_cast<const T*>(nullptr));
      }
    };
    template <class TType>
    struct LooperMaker : public ComponentMaker<edm::eventsetup::LooperMakerTraits, TType> {};
    typedef ComponentFactory<LooperMakerTraits> LooperFactory;

    typedef edmplugin::PluginFactory<edm::eventsetup::ComponentMakerBase<LooperMakerTraits>*()> LooperPluginFactory;
  }  // namespace eventsetup
}  // namespace edm

#define DEFINE_FWK_LOOPER(type)                                                                       \
  DEFINE_EDM_PLUGIN(edm::eventsetup::LooperPluginFactory, edm::eventsetup::LooperMaker<type>, #type); \
  DEFINE_DESC_FILLER_FOR_EDLOOPERS(type)

#endif
