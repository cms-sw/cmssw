#ifndef ServiceRegistry_ServiceRegistry_h
#define ServiceRegistry_ServiceRegistry_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServiceRegistry
//
/**\class ServiceRegistry ServiceRegistry.h FWCore/ServiceRegistry/interface/ServiceRegistry.h

 Description: Manages the 'thread specific' instance of Services

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 13:33:00 EDT 2005
//

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ServicesManager.h"

// system include files
#include <vector>

// forward declarations
namespace edm {
   class FwkImpl;
   namespace serviceregistry {
      template< typename T> class ServiceWrapper;
   }

   class ServiceRegistry {
   public:

      class Operate {
      public:
         Operate(ServiceToken const& iToken) :
         oldToken_(ServiceRegistry::instance().setContext(iToken)) {}
         ~Operate() {
            ServiceRegistry::instance().unsetContext(oldToken_);
         }

         //override operator new to stop use on heap?
      private:
         Operate(Operate const&) = delete; //stop default
         Operate const& operator=(Operate const&) = delete; //stop default
         ServiceToken oldToken_;
      };

      friend class edm::FwkImpl;
      friend int main(int argc, char* argv[]);
      friend class Operate;

      virtual ~ServiceRegistry();

      // ---------- const member functions ---------------------
      template<typename T>
      T& get() const {
         if(nullptr == manager_.get()) {
            Exception::throwThis(errors::NotFound,
              "Service"
              " no ServiceRegistry has been set for this thread");
         }
         return manager_-> template get<T>();
      }

      template<typename T>
      bool isAvailable() const {
         if(nullptr == manager_.get()) {
            Exception::throwThis(errors::NotFound,
              "Service"
              " no ServiceRegistry has been set for this thread");
         }
         return manager_-> template isAvailable<T>();
      }
      /** The token can be passed to another thread in order to have the
         same services available in the other thread.
         */
      ServiceToken presentToken() const;
      // ---------- static member functions --------------------
      static ServiceRegistry& instance();

      // ---------- member functions ---------------------------

      static ServiceToken createServicesFromConfig(std::string const& config);

   public: // Made public (temporarily) at the request of Emilio Meschi.
      static ServiceToken createSet(std::vector<ParameterSet>&);
      static ServiceToken createSet(std::vector<ParameterSet>&,
                                    ServiceToken,
                                    serviceregistry::ServiceLegacy,
                                    bool associate = true);
      /// create a service token that holds the service defined by iService
      template<typename T>
      static ServiceToken createContaining(std::unique_ptr<T> iService){
         std::vector<edm::ParameterSet> config;
         auto manager = std::make_shared<serviceregistry::ServicesManager>(config);
         auto wrapper = std::make_shared<serviceregistry::ServiceWrapper<T> >(std::move(iService));
         manager->put(wrapper);
         return manager;
      }
      template<typename T>
      static ServiceToken createContaining(std::unique_ptr<T> iService,
                                           ServiceToken iToken,
                                           serviceregistry::ServiceLegacy iLegacy){
         std::vector<edm::ParameterSet> config;
         auto manager = std::make_shared<serviceregistry::ServicesManager>(iToken, iLegacy, config);
         auto wrapper = std::make_shared<serviceregistry::ServiceWrapper<T> >(std::move(iService));
         manager->put(wrapper);
         return manager;
      }
      /// create a service token that holds the service held by iWrapper
      template<typename T>
      static ServiceToken createContaining(std::shared_ptr<serviceregistry::ServiceWrapper<T> > iWrapper) {
         std::vector<edm::ParameterSet> config;
         auto manager = std::make_shared<serviceregistry::ServicesManager>(config);
         manager->put(iWrapper);
         return manager;
      }
      template<typename T>
      static ServiceToken createContaining(std::shared_ptr<serviceregistry::ServiceWrapper<T> > iWrapper,
                                           ServiceToken iToken,
                                           serviceregistry::ServiceLegacy iLegacy){
         std::vector<edm::ParameterSet> config;
         auto manager = std::make_shared<serviceregistry::ServicesManager>(iToken, iLegacy, config);
         manager->put(iWrapper);
         return manager;
      }

private:
      //returns old token
      ServiceToken setContext(ServiceToken const& iNewToken);
      void unsetContext(ServiceToken const& iOldToken);

      ServiceRegistry();
      ServiceRegistry(ServiceRegistry const&); // stop default

      ServiceRegistry const& operator=(ServiceRegistry const&); // stop default

      // ---------- member data --------------------------------
      std::shared_ptr<serviceregistry::ServicesManager> manager_;
   };
}

#endif
