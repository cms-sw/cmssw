// -*- C++ -*-
#ifndef Framework_ComponentMaker_h
#define Framework_ComponentMaker_h
//
// Package:     Framework
// Class  :     ComponentMaker
//
/**\class edm::eventsetup::ComponentMaker

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

#include <format>

// user include files
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/ESProductResolverProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ComponentConstructionSentry.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
  namespace eventsetup {

    // forward declarations
    class EventSetupProvider;

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
      virtual std::shared_ptr<base_type> addTo(EventSetupProvider& iProvider, ParameterSet& iConfiguration) const = 0;
    };

    template <class T, class TComponent>
    class ComponentMaker : public ComponentMakerBase<T> {
    public:
      ComponentMaker() {}
      ComponentMaker(const ComponentMaker&) = delete;                   // stop default
      const ComponentMaker& operator=(const ComponentMaker&) = delete;  // stop default
      typedef typename T::base_type base_type;

      std::shared_ptr<base_type> addTo(EventSetupProvider& iProvider, ParameterSet& iConfiguration) const override;

    private:
      void setDescription(ESProductResolverProvider* iProv, const ComponentDescription& iDesc) const {
        iProv->setDescription(iDesc);
      }
      void setDescriptionForFinder(EventSetupRecordIntervalFinder* iFinder, const ComponentDescription& iDesc) const {
        iFinder->setDescriptionForFinder(iDesc);
      }
      void setPostConstruction(ESProductResolverProvider* iProv, const edm::ParameterSet& iPSet) const {
        //The 'appendToDataLabel' parameter was added very late in the development cycle and since
        // the ParameterSet is not sent to the base class we must set the value after construction
        iProv->setAppendToDataLabel(iPSet);
      }
      void setDescription(void*, const ComponentDescription&) const {}
      void setDescriptionForFinder(void*, const ComponentDescription&) const {}
      void setPostConstruction(void*, const edm::ParameterSet&) const {}
    };

    template <class T, class TComponent>
    std::shared_ptr<typename ComponentMaker<T, TComponent>::base_type> ComponentMaker<T, TComponent>::addTo(
        EventSetupProvider& iProvider, ParameterSet& iConfiguration) const {
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
          iException.addContext(std::format(
              "Validating configuration of {} of type {} with label: '{}'", T::baseType(), modtype, moduleLabel));
          throw;
        }
      }
      const ComponentDescription description = this->createComponentDescription(iConfiguration);
      std::shared_ptr<TComponent> component;
      {
        //here would be where the construction signal would go
        ComponentConstructionSentry sentry(iProvider, description);
        component = std::make_shared<TComponent>(iConfiguration);
      }

      this->setDescription(component.get(), description);
      this->setDescriptionForFinder(component.get(), description);
      this->setPostConstruction(component.get(), iConfiguration);

      T::addTo(iProvider, component);

      return component;
    }
  }  // namespace eventsetup
}  // namespace edm
#endif
