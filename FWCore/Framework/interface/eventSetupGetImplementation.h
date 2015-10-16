#ifndef Framework_eventSetupGetImplementation_h
#define Framework_eventSetupGetImplementation_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     eventSetupGetImplementation
// 
/**\class eventSetupGetImplementation eventSetupGetImplementation.h FWCore/Framework/interface/eventSetupGetImplementation.h

 Description: decleration of the template function that implements the EventSetup::get method

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Mar 25 16:31:17 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/HCMethods.h"
#include "FWCore/Framework/interface/NoRecordException.h"

namespace edm {
  class EventSetup;
   namespace eventsetup {
      template< class T>
      inline void eventSetupGetImplementation(EventSetup const& iEventSetup, T const*& iValue) {
         T const* temp = heterocontainer::find<EventSetupRecordKey, T const>(iEventSetup);
         if(0 == temp) {
            throw NoRecordException<T>(iovSyncValueFrom(iEventSetup), recordDoesExist(iEventSetup, EventSetupRecordKey::makeKey<T>()));
         }
         iValue = temp;
      }
     
     template< class T>
     inline void eventSetupTryToGetImplementation(EventSetup const& iEventSetup, T const*& iValue) {
       iValue = heterocontainer::find<EventSetupRecordKey, T const>(iEventSetup);
     }

   }
}

#endif
