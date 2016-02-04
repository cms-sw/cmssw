#ifndef ServiceRegistry_ServiceWrapper_h
#define ServiceRegistry_ServiceWrapper_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServiceWrapper
// 
/**\class ServiceWrapper ServiceWrapper.h FWCore/ServiceRegistry/interface/ServiceWrapper.h

 Description: Wrapper around a Service

 Usage:
    Implementation detail of the ServiceRegistry system

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 13:33:01 EDT 2005
//

// system include files
#include <memory>

// user include files
#include "FWCore/ServiceRegistry/interface/ServiceWrapperBase.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// forward declarations
namespace edm {
   class ParameterSet;
   class ActivityRegistry;

   namespace serviceregistry {

      template< class T>
      class ServiceWrapper : public ServiceWrapperBase
      {

public:
         ServiceWrapper(std::auto_ptr<T> iService) :
         service_(iService) {}
         //virtual ~ServiceWrapper();
         
         // ---------- const member functions ---------------------
         T const& get() const { return *service_; }
         
         // ---------- static member functions --------------------
         
         // ---------- member functions ---------------------------
         T& get() { return *service_; }

private:
         ServiceWrapper(const ServiceWrapper&); // stop default
         
         const ServiceWrapper& operator=(const ServiceWrapper&); // stop default
         
         // ---------- member data --------------------------------
         edm::propagate_const<std::auto_ptr<T>> service_;
         
      };
   }
}

#endif
