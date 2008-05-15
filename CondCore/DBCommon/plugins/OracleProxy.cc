#include "OracleProxy.h"
#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
cond::OracleProxy::OracleProxy(const std::string& userconnect):cond::TechnologyProxy(userconnect){
}
cond::OracleProxy::~OracleProxy(){
}
std::string 
cond::OracleProxy::getRealConnectString() const{
  return m_userconnect;
}
void 
cond::OracleProxy::setupSession(cond::DBSession& session){
}
DEFINE_EDM_PLUGIN(cond::TechnologyProxyFactory,cond::OracleProxy,"oracle");
