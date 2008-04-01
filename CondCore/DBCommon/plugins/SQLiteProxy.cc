#include "SQLiteProxy.h"
#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
#include "CondCore/DBCommon/interface/FipProtocolParser.h"
#include "SealKernel/Context.h"
#include "SealKernel/ComponentLoader.h"
#include "CondCore/DBCommon/interface/DBSession.h"
cond::SQLiteProxy::SQLiteProxy(const std::string& userconnect):cond::TechnologyProxy(userconnect){
}
cond::SQLiteProxy::~SQLiteProxy(){
}
std::string 
cond::SQLiteProxy::getRealConnectString( ) const{
  if( m_userconnect.find("sqlite_fip:") != std::string::npos ){
    cond::FipProtocolParser p;
    return p.getRealConnect(m_userconnect);
  }
  return m_userconnect;
}
void 
cond::SQLiteProxy::setupSession(cond::DBSession& session){
}
DEFINE_EDM_PLUGIN(cond::TechnologyProxyFactory,cond::SQLiteProxy,"sqlite");
