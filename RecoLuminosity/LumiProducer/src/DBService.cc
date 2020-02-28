#include "RecoLuminosity/LumiProducer/interface/DBService.h"
#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
#include "RelationalAccess/ConnectionService.h"
#include "CoralBase/Exception.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/AccessMode.h"

#include <iostream>
lumi::service::DBService::DBService(const edm::ParameterSet& iConfig)
    : m_svc(std::make_unique<coral::ConnectionService>()), m_dbconfig(std::make_unique<lumi::DBConfig>(*m_svc)) {
  std::string authpath = iConfig.getUntrackedParameter<std::string>("authPath", "");
  if (!authpath.empty()) {
    m_dbconfig->setAuthentication(authpath);
  }
}

lumi::service::DBService::~DBService() {}

lumi::service::ISessionProxyPtr lumi::service::DBService::connectReadOnly(const std::string& connectstring) {
  std::unique_lock<std::mutex> lock(m_mutex);

  return ISessionProxyPtr(std::unique_ptr<coral::ISessionProxy>(m_svc->connect(connectstring, coral::ReadOnly)),
                          std::move(lock));
}
