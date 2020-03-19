#ifndef RecoLuminosity_LumiProducer_DBService_h
#define RecoLuminosity_LumiProducer_DBService_h
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ConnectionService.h"

#include <string>
#include <mutex>
#include <memory>

namespace lumi {
  class DBConfig;
  namespace service {

    class ISessionProxyPtr {
    public:
      ISessionProxyPtr(std::unique_ptr<coral::ISessionProxy> iProxy, std::unique_lock<std::mutex> iLock)
          : m_lock(std::move(iLock)), m_proxy(std::move(iProxy)) {}

      coral::ISessionProxy* operator->() { return m_proxy.get(); }

    private:
      std::unique_lock<std::mutex> m_lock;
      std::unique_ptr<coral::ISessionProxy> m_proxy;
    };

    class DBService {
    public:
      DBService(const edm::ParameterSet& iConfig);
      ~DBService();

      ISessionProxyPtr connectReadOnly(const std::string& connectstring);

    private:
      std::unique_ptr<coral::ConnectionService> m_svc;
      std::unique_ptr<lumi::DBConfig> m_dbconfig;
      std::mutex m_mutex;
    };  //cl DBService
  }     // namespace service
}  // namespace lumi
#endif
