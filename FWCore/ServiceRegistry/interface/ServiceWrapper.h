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
// $Id: ServiceWrapper.h,v 1.3 2008/01/17 01:02:01 wmtan Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/ServiceRegistry/interface/ServiceWrapperBase.h"

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
         T& get() const { return *service_; }
         
         // ---------- static member functions --------------------
         
         // ---------- member functions ---------------------------

private:
         ServiceWrapper(const ServiceWrapper&); // stop default
         
         const ServiceWrapper& operator=(const ServiceWrapper&); // stop default
         
         // ---------- member data --------------------------------
         std::auto_ptr<T> service_;
         
      };
   }
}

#endif
