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
//

#include <typeinfo>

// forward declarations
namespace edm {
  class ParameterSet;
  class ActivityRegistry;

  namespace service {
    inline bool isProcessWideService(void const* /*service*/) { return false; }
  }  // namespace service

  namespace serviceregistry {
    class SaveConfiguration;
    class ServiceWrapperBase;
    class ServicesManager;

    class ServiceMakerBase {
    public:
      ServiceMakerBase();
      ServiceMakerBase(ServiceMakerBase const&) = delete;                   // stop default
      ServiceMakerBase const& operator=(ServiceMakerBase const&) = delete;  // stop default
      virtual ~ServiceMakerBase();

      // ---------- const member functions ---------------------
      virtual std::type_info const& serviceType() const = 0;

      virtual bool make(ParameterSet const&, ActivityRegistry&, ServicesManager&) const = 0;

      virtual bool processWideService() const = 0;

      virtual bool saveConfiguration() const = 0;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

    protected:
      bool testSaveConfiguration(SaveConfiguration const*) const { return true; }
      bool testSaveConfiguration(void const*) const { return false; }

    private:
      // ---------- member data --------------------------------
    };
  }  // namespace serviceregistry
}  // namespace edm

#endif
