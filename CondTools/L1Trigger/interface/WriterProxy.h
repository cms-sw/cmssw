#ifndef CondTools_L1Trigger_WriterProxy_h
#define CondTools_L1Trigger_WriterProxy_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondTools/L1Trigger/interface/Exception.h"

#include <string>

namespace l1t {

  /* This is class that is used to save data to DB. Saving requires that we should know types at compile time.
 * This means that I cannot create simple class that saves all records. So, I create a base class, and template
 * version of it, that will procede with saving. This approach is the same as used in DataProxy.
 */
  class WriterProxy {
  public:
    virtual ~WriterProxy() {}

    /* Saves record and type from given event setup to pool DB. This method should not worry
         * about such things as IOV and so on. It should return new payload token and then
         * the framework would take care of it.
         *
         * This method should not care of pool transactions and connections management.
         * In case some need other methods, like delete and update, one should add more abstract
         * methods here.
         */

    virtual std::string save(const edm::EventSetup& setup) const = 0;

  protected:
  };

  /* Concrete implementation of WriteProxy. This will do all the saving, also user of new types that should be saved
 * should instaciate a new version of this class and register it in plugin system.
 */
  template <class Record, class Type>
  class WriterProxyT : public WriterProxy {
  public:
    ~WriterProxyT() override {}

    /* This method requires that Record and Type supports copy constructor */
    std::string save(const edm::EventSetup& setup) const override {
      // load record and type from EventSetup and save them in db
      edm::ESHandle<Type> handle;

      try {
        setup.get<Record>().get(handle);
      } catch (l1t::DataAlreadyPresentException& ex) {
        return std::string();
      }

      // If handle is invalid, then data is already in DB

      edm::Service<cond::service::PoolDBOutputService> poolDb;
      if (!poolDb.isAvailable()) {
        throw cond::Exception("DataWriter: PoolDBOutputService not available.");
      }
      poolDb->forceInit();
      cond::persistency::Session session = poolDb->session();
      cond::persistency::TransactionScope tr(session.transaction());
      // if throw transaction will unroll
      ///	    tr.start(false);

      std::shared_ptr<Type> pointer = std::make_shared<Type>(*(handle.product()));
      std::string payloadToken = session.storePayload(*pointer);
      ///	    tr.commit();
      tr.close();
      return payloadToken;
    }
  };

  typedef edmplugin::PluginFactory<l1t::WriterProxy*()> WriterFactory;

// Defines new type, creates static instance of this class and register it for plugin
#define REGISTER_L1_WRITER(record, type)                            \
  template class l1t::WriterProxyT<record, type>;                   \
  typedef l1t::WriterProxyT<record, type> record##_##type##_Writer; \
  DEFINE_EDM_PLUGIN(l1t::WriterFactory, record##_##type##_Writer, #record "@" #type "@Writer")

}  // namespace l1t

#endif
