#ifndef Framework_ComponentMaker_h
#define Framework_ComponentMaker_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ComponentMaker
//
/**\class ComponentMaker ComponentMaker.h FWCore/Framework/interface/ComponentMaker.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Wed May 25 16:56:05 EDT 2005
//

// system include files
#include <memory>
#include <string>

#include <fmt/format.h>

// user include files
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"
#include "FWCore/Utilities/interface/ConvertException.h"

// forward declarations

namespace edm {
  namespace eventsetup {
    class EventSetupProvider;
    class EventSetupsController;
    class DataProxyProvider;

    class ComponentMakerBaseHelper {
    public:
      virtual ~ComponentMakerBaseHelper() {}

    protected:
      ComponentDescription createComponentDescription(ParameterSet const& iConfiguration) const;
    };

    template <class T>
    class ComponentMakerBase : public ComponentMakerBaseHelper {
    public:
      typedef typename T::base_type base_type;
      virtual std::shared_ptr<base_type> addTo(EventSetupsController& esController,
                                               EventSetupProvider& iProvider,
                                               ParameterSet& iConfiguration,
                                               bool replaceExisting) const = 0;
    };

    template <class T, class TComponent>
    class ComponentMaker : public ComponentMakerBase<T> {
    public:
      ComponentMaker() {}
      ComponentMaker(const ComponentMaker&) = delete;                   // stop default
      const ComponentMaker& operator=(const ComponentMaker&) = delete;  // stop default
      typedef typename T::base_type base_type;

      // ---------- const member functions ---------------------
      std::shared_ptr<base_type> addTo(EventSetupsController& esController,
                                       EventSetupProvider& iProvider,
                                       ParameterSet& iConfiguration,
                                       bool replaceExisting) const override;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
    private:
      void setDescription(DataProxyProvider* iProv, const ComponentDescription& iDesc) const {
        iProv->setDescription(iDesc);
      }
      void setDescriptionForFinder(EventSetupRecordIntervalFinder* iFinder, const ComponentDescription& iDesc) const {
        iFinder->setDescriptionForFinder(iDesc);
      }
      void setPostConstruction(DataProxyProvider* iProv, const edm::ParameterSet& iPSet) const {
        //The 'appendToDataLabel' parameter was added very late in the development cycle and since
        // the ParameterSet is not sent to the base class we must set the value after construction
        iProv->setAppendToDataLabel(iPSet);
      }
      void setDescription(void*, const ComponentDescription&) const {}
      void setDescriptionForFinder(void*, const ComponentDescription&) const {}
      void setPostConstruction(void*, const edm::ParameterSet&) const {}
      // ---------- member data --------------------------------
    };

    template <class T, class TComponent>
    std::shared_ptr<typename ComponentMaker<T, TComponent>::base_type> ComponentMaker<T, TComponent>::addTo(
        EventSetupsController& esController,
        EventSetupProvider& iProvider,
        ParameterSet& iConfiguration,
        bool replaceExisting) const {
      // This adds components to the EventSetupProvider for the process. It might
      // make a new component then add it or reuse a component from an earlier
      // SubProcess or the top level process and add that.

      {
        auto modtype = iConfiguration.getParameter<std::string>("@module_type");
        auto moduleLabel = iConfiguration.getParameter<std::string>("@module_label");
        try {
          edm::convertException::wrap([&]() {
            ConfigurationDescriptions descriptions(T::baseType(), modtype);
            fillDetails::fillIfExists<TComponent>(descriptions);
            fillDetails::prevalidateIfExists<TComponent>(descriptions);
            descriptions.validate(iConfiguration, moduleLabel);
            iConfiguration.registerIt();
          });
        } catch (cms::Exception& iException) {
          iException.addContext(fmt::format(
              "Validating configuration of {} of type {} with label: '{}'", T::baseType(), modtype, moduleLabel));
          throw;
        }
      }

      if (!replaceExisting) {
        std::shared_ptr<typename T::base_type> alreadyMadeComponent =
            T::getComponentAndRegisterProcess(esController, iConfiguration);

        if (alreadyMadeComponent) {
          // This is for the case when a component is shared between
          // a SubProcess and a previous SubProcess or the top level process
          // because the component has an identical configuration to a component
          // from the top level process or earlier SubProcess.
          std::shared_ptr<TComponent> component(
              std::static_pointer_cast<TComponent, typename T::base_type>(alreadyMadeComponent));
          T::addTo(iProvider, component, iConfiguration, true);
          return component;
        }
      }

      std::shared_ptr<TComponent> component = std::make_shared<TComponent>(iConfiguration);
      ComponentDescription description = this->createComponentDescription(iConfiguration);

      this->setDescription(component.get(), description);
      this->setDescriptionForFinder(component.get(), description);
      this->setPostConstruction(component.get(), iConfiguration);

      if (replaceExisting) {
        // This case is for ESProducers where in the first pass
        // the algorithm thought the component could be shared
        // across SubProcess's because there was an ESProducer
        // from a previous process with an identical configuration.
        // But in a later check it was determined that sharing was not
        // possible because other components associated with the
        // same record or records that record depends on had
        // differing configurations.
        T::replaceExisting(iProvider, component);
      } else {
        // This is for the case when a new component is being constructed.
        // All components for the top level process fall in this category.
        // Or it could be a SubProcess where neither the top level process
        // nor any prior SubProcess had a component with exactly the same configuration.
        T::addTo(iProvider, component, iConfiguration, false);
        T::putComponent(esController, iConfiguration, component);
      }
      return component;
    }
  }  // namespace eventsetup
}  // namespace edm
#endif
