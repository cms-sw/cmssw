#ifndef FWCore_Framework_SourceFactory_h
#define FWCore_Framework_SourceFactory_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     SourceFactory
// 
/**\class SourceFactory SourceFactory.h FWCore/Framework/interface/SourceFactory.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Wed May 25 18:01:38 EDT 2005
//

// system include files
#include <string>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ComponentFactory.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"

// forward declarations

namespace edm {
   class EventSetupRecordIntervalFinder;
   class ParameterSet;

   namespace eventsetup {
      class DataProxyProvider;
      class EventSetupsController;
      
      template<class T>
         void addProviderTo(EventSetupProvider& iProvider, boost::shared_ptr<T> iComponent, const DataProxyProvider*) 
      {
            boost::shared_ptr<DataProxyProvider> pProvider(iComponent);
            ComponentDescription description = pProvider->description();
            description.isSource_=true;
            pProvider->setDescription(description);
            iProvider.add(pProvider);
      }
      template<class T>
         void addProviderTo(EventSetupProvider& /* iProvider */, boost::shared_ptr<T> /*iComponent*/, const void*) 
      {
            //do nothing
      }
      
      struct SourceMakerTraits {
         typedef EventSetupRecordIntervalFinder base_type;
         static std::string name();
         template<class T>
         static void addTo(EventSetupProvider& iProvider,
                           boost::shared_ptr<T> iComponent,
                           ParameterSet const& iConfiguration,
                           bool matchesPreceding)
            {
               if (matchesPreceding) {
                  logInfoWhenSharing(iConfiguration);
               }
               //a source does not always have to be a provider
               addProviderTo(iProvider, iComponent, static_cast<const T*>(nullptr));
               boost::shared_ptr<EventSetupRecordIntervalFinder> pFinder(iComponent);
               iProvider.add(pFinder);
            }
         static void replaceExisting(EventSetupProvider& iProvider, boost::shared_ptr<EventSetupRecordIntervalFinder> iComponent); 
               
         static boost::shared_ptr<base_type> getComponentAndRegisterProcess(EventSetupsController& esController,
                                                                            ParameterSet const& iConfiguration);

         static void putComponent(EventSetupsController& esController,
                                  ParameterSet const& iConfiguration,
                                  boost::shared_ptr<base_type> const& component);

         static void logInfoWhenSharing(ParameterSet const& iConfiguration);
      };

      template< class TType>
         struct SourceMaker : public ComponentMaker<edm::eventsetup::SourceMakerTraits,TType> {};
      typedef  ComponentFactory<SourceMakerTraits> SourceFactory ;
      
      typedef edmplugin::PluginFactory<edm::eventsetup::ComponentMakerBase<edm::eventsetup::SourceMakerTraits>* ()> SourcePluginFactory;
   }
}

#define DEFINE_FWK_EVENTSETUP_SOURCE(type) \
DEFINE_EDM_PLUGIN (edm::eventsetup::SourcePluginFactory,edm::eventsetup::SourceMaker<type>,#type); \
DEFINE_DESC_FILLER_FOR_ESSOURCES(type)

#endif
