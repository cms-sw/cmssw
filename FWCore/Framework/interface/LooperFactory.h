#ifndef FWCore_Framework_LooperFactory_h
#define FWCore_Framework_LooperFactory_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     LooperFactory
// 
/**\class LooperFactory LooperFactory.h FWCore/Framework/interface/LooperFactory.h

 Description: PluginManager factory for creating EDLoopers

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Wed May 25 18:01:38 EDT 2005
//

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/ComponentFactory.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"

// forward declarations
namespace edm {
   class EDLooper;
   class EventSetupRecordIntervalFinder;

   namespace eventsetup {
      class DataProxyProvider;
      namespace looper {
      template<class T>
         void addProviderTo(EventSetupProvider& iProvider, boost::shared_ptr<T> iComponent, const DataProxyProvider*) 
      {
            boost::shared_ptr<DataProxyProvider> pProvider(iComponent);
            ComponentDescription description = pProvider->description();
            description.isSource_=true;
            description.isLooper_=true;
	    if(description.label_ =="@main_looper") {
	       //remove the 'hidden' label so that es_prefer statements will work
	       description.label_ ="";
	    }
            pProvider->setDescription(description);
            iProvider.add(pProvider);
      }
      template<class T>
         void addProviderTo(EventSetupProvider& /* iProvider */, boost::shared_ptr<T> /*iComponent*/, const void*) 
      {
            //do nothing
      }

      template<class T>
        void addFinderTo(EventSetupProvider& iProvider, boost::shared_ptr<T> iComponent, const EventSetupRecordIntervalFinder*) 
      {
          boost::shared_ptr<EventSetupRecordIntervalFinder> pFinder(iComponent);

          ComponentDescription description = pFinder->descriptionForFinder();
          description.isSource_=true;
          description.isLooper_=true;
          if(description.label_ =="@main_looper") {
            //remove the 'hidden' label so that es_prefer statements will work
            description.label_ ="";
          }
          pFinder->setDescriptionForFinder(description);
          
          iProvider.add(pFinder);
      }
      template<class T>
        void addFinderTo(EventSetupProvider& /* iProvider */, boost::shared_ptr<T> /*iComponent*/, const void*) 
      {
          //do nothing
      }
      }
      struct LooperMakerTraits {
         typedef EDLooper base_type;
         static std::string name();
         template<class T>
            static void addTo(EventSetupProvider& iProvider, boost::shared_ptr<T> iComponent)
            {
               //a looper does not always have to be a provider or a finder
               looper::addProviderTo(iProvider, iComponent, static_cast<const T*>(0));
               looper::addFinderTo(iProvider, iComponent, static_cast<const T*>(0));
            }
               
      };
      template< class TType>
         struct LooperMaker : public ComponentMaker<edm::eventsetup::LooperMakerTraits,TType> {};
      typedef  ComponentFactory<LooperMakerTraits> LooperFactory ;
      
      typedef edmplugin::PluginFactory<edm::eventsetup::ComponentMakerBase<LooperMakerTraits>* ()> LooperPluginFactory;
   }
}

#define DEFINE_FWK_LOOPER(type) \
DEFINE_EDM_PLUGIN (edm::eventsetup::LooperPluginFactory,edm::eventsetup::LooperMaker<type>,#type)

#endif
