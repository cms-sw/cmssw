#include "OracleProxy.h"
#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
cond::OracleProxy::OracleProxy(){
}
cond::OracleProxy::~OracleProxy(){
}
std::string 
cond::OracleProxy::getRealConnectString( const std::string& iValue ) const{
  return iValue;
}
void 
cond::OracleProxy::setupSession(){
}
void 
cond::OracleProxy::prepareConnection(){
}
void 
cond::OracleProxy::prepareTransaction(){
}
DEFINE_EDM_PLUGIN(cond::TechnologyProxyFactory,cond::OracleProxy,"oracle");
