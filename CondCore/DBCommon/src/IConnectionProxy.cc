#include "CondCore/DBCommon/interface/IConnectionProxy.h"
#include "RelationalAccess/IConnectionService.h"

//#include <iostream>
cond::IConnectionProxy::IConnectionProxy(
      coral::IConnectionService* connectionServiceHandle,
      const std::string& con,
      int connectionTimeOut,
      int idleConnectionCleanupPeriod):
  m_connectionSvcHandle(connectionServiceHandle),
  m_con(con),
  m_connectionTimeOut(connectionTimeOut),
  m_idleConnectionCleanupPeriod(idleConnectionCleanupPeriod){}
cond::IConnectionProxy::~IConnectionProxy(){}
int
cond::IConnectionProxy::connectionTimeOut() const{
  return m_connectionTimeOut;
}
std::string 
cond::IConnectionProxy::connectStr() const{
  return m_con;
}
