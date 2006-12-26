#ifndef Framework_SourceFactory_h
#define Framework_SourceFactory_h
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
// $Id: SourceFactory.h,v 1.16 2006/10/30 23:07:53 wmtan Exp $
//

// system include files
#include <string>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ComponentFactory.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"

// forward declarations

namespace edm {
   class EventSetupRecordIntervalFinder;

   namespace eventsetup {
      class DataProxyProvider;
      
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
            static void addTo(EventSetupProvider& iProvider, boost::shared_ptr<T> iComponent)
            {
               //a source does not always have to be a provider
               addProviderTo(iProvider, iComponent, static_cast<const T*>(0));
               boost::shared_ptr<EventSetupRecordIntervalFinder> pFinder(iComponent);
               iProvider.add(pFinder);
            }
               
      };
      template< class TType>
         struct SourceMaker : public ComponentMaker<edm::eventsetup::SourceMakerTraits,TType> {};
      typedef  ComponentFactory<SourceMakerTraits> SourceFactory ;
   }
}

#define DEFINE_FWK_EVENTSETUP_SOURCE(type) \
DEFINE_SEAL_MODULE (); \
DEFINE_SEAL_PLUGIN (edm::eventsetup::SourceFactory,edm::eventsetup::SourceMaker<type>,#type)

#define DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(type) \
DEFINE_SEAL_PLUGIN (edm::eventsetup::SourceFactory,edm::eventsetup::SourceMaker<type>,#type)

#endif
