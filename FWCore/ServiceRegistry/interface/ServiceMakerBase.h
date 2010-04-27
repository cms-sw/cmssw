#ifndef FWCore_ServiceRegistry_ServiceMakerBase_h
#define FWCore_ServiceRegistry_ServiceMakerBase_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServiceMakerBase
// 
/**\class ServiceMakerBase ServiceMakerBase.h FWCore/ServiceRegistry/interface/ServiceMakerBase.h

 Description: Base class for Service Makers

 Usage:
    Internal detail of implementation of the ServiceRegistry system

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 13:33:00 EDT 2005
// $Id: ServiceMakerBase.h,v 1.3 2007/06/14 02:25:50 wmtan Exp $
//

// system include files

// user include files

// forward declarations
namespace edm {
   class ParameterSet;
   class ActivityRegistry;
   
   namespace serviceregistry {

      class SaveConfiguration;
      class ServiceWrapperBase;
      class ServicesManager;
      
      class ServiceMakerBase {
         
public:
         ServiceMakerBase();
         virtual ~ServiceMakerBase();
         
         // ---------- const member functions ---------------------
         virtual const std::type_info& serviceType() const = 0;
      
         virtual bool make(const edm::ParameterSet&,
                           edm::ActivityRegistry&,
                           ServicesManager&) const = 0;

         virtual bool saveConfiguration() const = 0;
         
         // ---------- static member functions --------------------
         
         // ---------- member functions ---------------------------
         
protected:
         bool testSaveConfiguration(const SaveConfiguration*) const {return true;}
         bool testSaveConfiguration(const void*) const {return false;}

private:
         ServiceMakerBase(const ServiceMakerBase&); // stop default
         
         const ServiceMakerBase& operator=(const ServiceMakerBase&); // stop default
         
         // ---------- member data --------------------------------
         
      };
   }
}

#endif
