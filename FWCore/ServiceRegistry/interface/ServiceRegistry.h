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
// $Id: ServiceRegistry.h,v 1.2 2005/09/08 09:06:00 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServicesManager.h"

// forward declarations
namespace edm {
   class FwkImpl;
   class ServiceRegistry
   {

   public:
      
      class Operate {
        public:
         Operate(const ServiceToken& iToken) : 
         oldToken_( ServiceRegistry::instance().setContext( iToken ) )
         {}
         ~Operate() {
            ServiceRegistry::instance().unsetContext( oldToken_ );
         }
         
         //override operator new to stop use on heap?
        private:
         Operate(const Operate&); //stop default
         const Operate& operator=( const Operate&); //stop default
         ServiceToken oldToken_;
      };
      
      friend class edm::FwkImpl;
      friend int main(int argc, char* argv[]);
      friend class Operate;

      virtual ~ServiceRegistry();

      // ---------- const member functions ---------------------
      template<class T>
         T& get() const {
            if(0 == manager_.get()) {
               throw edm::Exception(edm::errors::NotFound,"Service")
               <<" no ServiceRegistry has been set for this thread";
            }
            return manager_-> template get<T>();
         }
      
      /** The token can be passed to another thread in order to have the
         same services available in the other thread.
         */
      ServiceToken presentToken() const;
      // ---------- static member functions --------------------
      static ServiceRegistry& instance();
      
      // ---------- member functions ---------------------------
      
   
   private:

      static ServiceToken createSet(const std::vector<ParameterSet>&);
      static ServiceToken createSet(const std::vector<ParameterSet>&,
                                    ServiceToken,serviceregistry::ServiceLegacy);
      
      //returns old token
      ServiceToken setContext(const ServiceToken& iNewToken);
      void unsetContext(const ServiceToken& iOldToken);
      
      ServiceRegistry();
      ServiceRegistry(const ServiceRegistry&); // stop default

      const ServiceRegistry& operator=(const ServiceRegistry&); // stop default

      // ---------- member data --------------------------------
      boost::shared_ptr<serviceregistry::ServicesManager> manager_;
   };
}

#endif
