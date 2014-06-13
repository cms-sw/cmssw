#ifndef ServiceRegistry_ServiceToken_h
#define ServiceRegistry_ServiceToken_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServiceToken
// 
/**\class ServiceToken ServiceToken.h FWCore/ServiceRegistry/interface/ServiceToken.h

 Description: Token used to denote a 'service set' 

 Usage:
    When you request a new 'service set' to be created from the ServiceRegistry, 
  the ServiceRegistry will return a ServiceToken.  When you want this 'service set' to be used,
  create a ServiceRegistry::Operate by passing the ServiceToken via the constructor.

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Sep  6 18:31:44 EDT 2005
//

// system include files
#include <memory>

// user include files

// forward declarations
namespace edm {
   class ServiceRegistry;
   class ActivityRegistry;
   
   namespace serviceregistry {
      class ServicesManager;
   }
   
   class ServiceToken
   {
      friend class edm::ServiceRegistry;
      friend class edm::serviceregistry::ServicesManager;
    public:
      ServiceToken() {}
      //virtual ~ServiceToken();
      
      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      ///the argument's signals are propagated to the Service's held by the token
      void connectTo(ActivityRegistry&);
      ///the argument's signals will forward the token's signals
      void connect(ActivityRegistry&);

      ///copy our Service's slots to the argument's signals
      void copySlotsTo(ActivityRegistry&);
      ///the copy the argument's slots to the token's signals
      void copySlotsFrom(ActivityRegistry&);
      
    private:
      ServiceToken(std::shared_ptr<edm::serviceregistry::ServicesManager>  iManager):
      manager_(iManager) {}
      
      //ServiceToken(const ServiceToken&); // stop default

      //const ServiceToken& operator=(const ServiceToken&); // stop default

      // ---------- member data --------------------------------
      std::shared_ptr<edm::serviceregistry::ServicesManager> manager_;
   };
}

#endif
