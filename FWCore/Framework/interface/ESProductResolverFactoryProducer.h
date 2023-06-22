#ifndef Framework_ESProductResolverFactoryProducer_h
#define Framework_ESProductResolverFactoryProducer_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProductResolverFactoryProducer
//
/**\class ESProductResolverFactoryProducer ESProductResolverFactoryProducer.h FWCore/Framework/interface/ESProductResolverFactoryProducer.h

 Description: An EventSetup algorithmic Provider that manages Factories of Proxies

 Usage:
    This class is used when the algorithms in the EventSetup that are to be run on demand are encapsulated
  in edm::eventsetup::ProductResolvers's.  This 'design pattern' is more flexible than having the algorithm embedded
  directly in the Provider (see ESProducer for such an implemenation).

    Users inherit from this class and then call the 'registerFactory' method in their class' constructor
  in order to get their Proxies registered.  For most users, the already available templated Factory classes
  should suffice and therefore they should not need to create their own Factories.

Example: register one Factory that creates a resolver that takes no arguments
\code
   class FooResolver : public edm::eventsetup::ESProductResolver { ... };
   class FooProd : public edm::ESProductResolverFactoryProducer { ... };

   FooProd::FooProd(const edm::ParameterSet&) {
      typedef edm::eventsetup::ESProductResolverFactoryTemplate<FooResolver> > TYPE;
      registerFactory(std::make_unique<TYPE>();
   };
   
\endcode

Example: register one Factory that creates a resolver that takes one argument
\code
class BarResolver : public edm::eventsetup::ESProductResolver { ...
   BarResolver(const edm::ParameterSet&) ;
   ... };
class BarProd : public edm::ESProductResolverFactoryProducer { ... };

BarProd::BarProd(const edm::ParameterSet& iPS) {
   typedef edm::eventsetup::ESProductResolverArgumentFactoryTemplate<BarResolver, edm::ParmeterSet> TYPE;
   registerFactory(std::make_unique<TYPE>(iPS);
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

// user include files

// forward declarations
#include "FWCore/Framework/interface/ESProductResolverProvider.h"
#include "FWCore/Utilities/interface/propagate_const.h"

namespace edm {
  namespace eventsetup {
    class ESProductResolverFactoryBase;

    struct FactoryInfo {
      FactoryInfo() : key_(), factory_() {}
      FactoryInfo(const DataKey& iKey, std::shared_ptr<ESProductResolverFactoryBase> iFactory) : key_(iKey), factory_(iFactory) {}
      DataKey key_;
      edm::propagate_const<std::shared_ptr<ESProductResolverFactoryBase>> factory_;
    };
  }  // namespace eventsetup

  class ESProductResolverFactoryProducer : public eventsetup::ESProductResolverProvider {
  public:
    ESProductResolverFactoryProducer();

    ESProductResolverFactoryProducer(const ESProductResolverFactoryProducer&) = delete;
    const ESProductResolverFactoryProducer& operator=(const ESProductResolverFactoryProducer&) = delete;

    ~ESProductResolverFactoryProducer() noexcept(false) override;

  protected:
    using EventSetupRecordKey = eventsetup::EventSetupRecordKey;

    KeyedResolversVector registerProxies(const EventSetupRecordKey&, unsigned int iovIndex) override;

    /** \param iFactory unique_ptr holding a new instance of a Factory
         \param iLabel extra string label used to get data (optional)
         Producer takes ownership of the Factory and uses it create the appropriate
         Resolver which is then registered with the EventSetup.  If used, this method should
         be called in inheriting class' constructor.
      */
    template <class TFactory>
    void registerFactory(std::unique_ptr<TFactory> iFactory, const std::string& iLabel = std::string()) {
      std::unique_ptr<eventsetup::ESProductResolverFactoryBase> temp(iFactory.release());
      registerFactoryWithKey(EventSetupRecordKey::makeKey<typename TFactory::RecordType>(), std::move(temp), iLabel);
    }

    virtual void registerFactoryWithKey(const EventSetupRecordKey& iRecord,
                                        std::unique_ptr<eventsetup::ESProductResolverFactoryBase> iFactory,
                                        const std::string& iLabel = std::string());

  private:
    // ---------- member data --------------------------------
    std::multimap<EventSetupRecordKey, eventsetup::FactoryInfo> record2Factories_;
  };

}  // namespace edm

#endif
