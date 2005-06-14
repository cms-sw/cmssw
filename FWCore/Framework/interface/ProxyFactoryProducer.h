#ifndef EVENTSETUPPRODUCER_PROXYFACTORYPRODUCER_H
#define EVENTSETUPPRODUCER_PROXYFACTORYPRODUCER_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     ProxyFactoryProducer
// 
/**\class ProxyFactoryProducer ProxyFactoryProducer.h Core/CoreFramework/interface/ProxyFactoryProducer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Apr  7 17:14:58 CDT 2005
// $Id: ProxyFactoryProducer.h,v 1.1 2005/05/29 02:29:53 wmtan Exp $
//

// system include files
#include <map>
#include "boost/shared_ptr.hpp"

// user include files

// forward declarations
#include "FWCore/CoreFramework/interface/DataProxyProvider.h"

namespace edm {
   namespace eventsetup {
      class ProxyFactoryBase;
      
      struct FactoryInfo {
         FactoryInfo() {}
         FactoryInfo( const DataKey& iKey, 
                      boost::shared_ptr<ProxyFactoryBase> iFactory )
         : key_( iKey ), 
         factory_( iFactory ) {} 
         DataKey key_;
         boost::shared_ptr<ProxyFactoryBase> factory_;
      };
      
class ProxyFactoryProducer : public DataProxyProvider
      
{

   public:
      ProxyFactoryProducer();
      virtual ~ProxyFactoryProducer();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void newInterval( const EventSetupRecordKey& iRecordType,
                                const ValidityInterval& iInterval ) ;

   protected:
      virtual void registerProxies( const EventSetupRecordKey& iRecord ,
                                    KeyedProxies& aProxyList ) ;

      template< class TFactory>
         void registerFactory(std::auto_ptr<TFactory> iFactory ) {
            std::auto_ptr<ProxyFactoryBase> temp( iFactory.release() );
            registerFactoryWithKey( 
               EventSetupRecordKey::makeKey<typename TFactory::record_type>(),
                                    temp);
         }
      template< class TFactory>
         void registerFactory(TFactory* iFactory ) {
            std::auto_ptr<TFactory> temp( iFactory);
            registerFactory( temp );
         }
      
   private:
      ProxyFactoryProducer( const ProxyFactoryProducer& ); // stop default

      const ProxyFactoryProducer& operator=( const ProxyFactoryProducer& ); // stop default

      virtual void registerFactoryWithKey( const EventSetupRecordKey& iRecord ,
                                          std::auto_ptr<ProxyFactoryBase>& iFactory );
      
      // ---------- member data --------------------------------
      std::multimap< EventSetupRecordKey, FactoryInfo > record2Factories_;

};

   }
}

#endif /* EVENTSETUPPRODUCER_PROXYFACTORYPRODUCER_H */
