#ifndef Framework_LooperFactory_h
#define Framework_LooperFactory_h
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
// $Id: LooperFactory.h,v 1.2 2006/10/26 20:38:09 wmtan Exp $
//

// system include files
#include <string>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ComponentFactory.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"

#include "FWCore/Utilities/interface/GCCPrerequisite.h"

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
   }
}

#if GCC_PREREQUISITE(3,4,4)

#define DEFINE_FWK_LOOPER(type) \
DEFINE_SEAL_MODULE (); \
DEFINE_SEAL_PLUGIN (edm::eventsetup::LooperFactory,edm::eventsetup::LooperMaker<type>,#type)

#define DEFINE_ANOTHER_FWK_LOOPER(type) \
DEFINE_SEAL_PLUGIN (edm::eventsetup::LooperFactory,edm::eventsetup::LooperMaker<type>,#type)

#else

#define DEFINE_FWK_LOOPER(type) \
DEFINE_SEAL_MODULE (); \
DEFINE_SEAL_PLUGIN (edm::eventsetup::LooperFactory,edm::eventsetup::LooperMaker<type>,#type);

#define DEFINE_ANOTHER_FWK_LOOPER(type) \
DEFINE_SEAL_PLUGIN (edm::eventsetup::LooperFactory,edm::eventsetup::LooperMaker<type>,#type);

#endif

#endif
