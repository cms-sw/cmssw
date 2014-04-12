#ifndef Framework_ESProxyFactoryProducer_h
#define Framework_ESProxyFactoryProducer_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProxyFactoryProducer
// 
/**\class ESProxyFactoryProducer ESProxyFactoryProducer.h FWCore/Framework/interface/ESProxyFactoryProducer.h

 Description: An EventSetup algorithmic Provider that manages Factories of Proxies

 Usage:
    This class is used when the algorithms in the EventSetup that are to be run on demand are encapsulated
  in edm::eventsetup::Proxy's.  This 'design pattern' is more flexible than having the algorithm embedded 
  directly in the Provider (see ESProducer for such an implemenation).

    Users inherit from this class and then call the 'registerFactory' method in their class' constructor
  in order to get their Proxies registered.  For most users, the already available templated Factory classes
  should suffice and therefore they should not need to create their own Factories.

Example: register one Factory that creates a proxy that takes no arguments
\code
   class FooProxy : public edm::eventsetup::DataProxy { ... };
   class FooProd : public edm::ESProxyFactoryProducer { ... };

   FooProd::FooProd(const edm::ParameterSet&) {
      typedef edm::eventsetup::ProxyFactoryTemplate<FooProxy> > TYPE;
      registerFactory(std::auto_ptr<TYPE>(new TYPE());
   };
   
\endcode

Example: register one Factory that creates a proxy that takes one argument
\code
class BarProxy : public edm::eventsetup::DataProxy { ...
   BarProxy(const edm::ParameterSet&) ;
   ... };
class BarProd : public edm::ESProxyFactoryProducer { ... };

BarProd::BarProd(const edm::ParameterSet& iPS) {
   typedef edm::eventsetup::ProxyArgumentFactoryTemplate<FooProxy, edm::ParmeterSet> TYPE;
   registerFactory(std::auto_ptr<TYPE>(new TYPE(iPS));
};

\endcode

*/
//
// Author:      Chris Jones
// Created:     Thu Apr  7 17:14:58 CDT 2005
//

// system include files
#include <map>
#include <memory>
#include <string>
#include "boost/shared_ptr.hpp"

// user include files

// forward declarations
#include "FWCore/Framework/interface/DataProxyProvider.h"

namespace edm {
   namespace eventsetup {
      class ProxyFactoryBase;
      
      struct FactoryInfo {
         FactoryInfo() : key_(), factory_() {}
         FactoryInfo(const DataKey& iKey, 
                      boost::shared_ptr<ProxyFactoryBase> iFactory)
         : key_(iKey), 
         factory_(iFactory) {} 
         DataKey key_;
         boost::shared_ptr<ProxyFactoryBase> factory_;
      };
   }      
   
class ESProxyFactoryProducer : public eventsetup::DataProxyProvider
{

   public:
      ESProxyFactoryProducer();
      virtual ~ESProxyFactoryProducer();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      ///overrides DataProxyProvider method
      virtual void newInterval(const eventsetup::EventSetupRecordKey& iRecordType,
                                const ValidityInterval& iInterval) ;

   protected:
      ///override DataProxyProvider method
      virtual void registerProxies(const eventsetup::EventSetupRecordKey& iRecord ,
                                    KeyedProxies& aProxyList) ;

      /** \param iFactory auto_ptr holding a new instance of a Factory
         \param iLabel extra string label used to get data (optional)
         Producer takes ownership of the Factory and uses it create the appropriate
         Proxy which is then registered with the EventSetup.  If used, this method should
         be called in inheriting class' constructor.
      */
      template< class TFactory>
         void registerFactory(std::auto_ptr<TFactory> iFactory,
                              const std::string& iLabel = std::string()) {
            std::auto_ptr<eventsetup::ProxyFactoryBase> temp(iFactory.release());
            registerFactoryWithKey(
                                   eventsetup::EventSetupRecordKey::makeKey<typename TFactory::record_type>(),
                                   temp,
                                   iLabel);
         }
      
      virtual void registerFactoryWithKey(const eventsetup::EventSetupRecordKey& iRecord ,
                                          std::auto_ptr<eventsetup::ProxyFactoryBase>& iFactory,
                                          const std::string& iLabel= std::string() );

   private:
      ESProxyFactoryProducer(const ESProxyFactoryProducer&); // stop default

      const ESProxyFactoryProducer& operator=(const ESProxyFactoryProducer&); // stop default

      
      // ---------- member data --------------------------------
      std::multimap< eventsetup::EventSetupRecordKey, eventsetup::FactoryInfo > record2Factories_;

};

}

#endif
