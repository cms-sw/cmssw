#ifndef EVENTSETUP_SOURCEFACTORY_H
#define EVENTSETUP_SOURCEFACTORY_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     SourceFactory
// 
/**\class SourceFactory SourceFactory.h Core/CoreFramework/interface/SourceFactory.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Wed May 25 18:01:38 EDT 2005
// $Id: SourceFactory.h,v 1.3 2005/05/26 20:41:11 chrjones Exp $
//

// system include files
#include <string>

// user include files
#include "FWCore/CoreFramework/interface/ComponentFactory.h"
#include "FWCore/CoreFramework/interface/EventSetupProvider.h"
// forward declarations

namespace edm {
   namespace eventsetup {
      class DataProxyProvider;
      class EventSetupRecordIntervalFinder;
      
      template<class T>
         void addProviderTo( EventSetupProvider& iProvider, boost::shared_ptr<T> iComponent, const DataProxyProvider*) 
      {
            boost::shared_ptr<DataProxyProvider> pProvider( iComponent );
            iProvider.add( pProvider );
      }
      template<class T>
         void addProviderTo( EventSetupProvider& iProvider, boost::shared_ptr<T> iComponent, const void*) 
      {
            //do nothing
      }
      
      struct SourceMakerTraits {
         static std::string name();
         template<class T>
            static void addTo( EventSetupProvider& iProvider, boost::shared_ptr<T> iComponent )
            {
               //a source does not always have to be a provider
               addProviderTo( iProvider, iComponent, static_cast<const T*>(0) );
               boost::shared_ptr<EventSetupRecordIntervalFinder> pFinder( iComponent );
               iProvider.add( pFinder );
            }
               
      };
      template< class TType>
         struct SourceMaker : public ComponentMaker<edm::eventsetup::SourceMakerTraits,TType> {};
      typedef  ComponentFactory<SourceMakerTraits> SourceFactory ;
   }
}

#define DEFINE_FWK_EVENTSETUP_SOURCE(type) \
DEFINE_SEAL_MODULE (); \
DEFINE_SEAL_PLUGIN (edm::eventsetup::SourceFactory,edm::eventsetup::SourceMaker<type>,#type);

#endif /* EVENTSETUP_SOURCEFACTORY_H */
