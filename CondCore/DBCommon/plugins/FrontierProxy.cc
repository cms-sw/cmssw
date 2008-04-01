#include "FrontierProxy.h"
#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
cond::FrontierProxy::FrontierProxy(){
}
cond::FrontierProxy::~FrontierProxy(){
}
std::string 
cond::FrontierProxy::getRealConnectString( const std::string& iValue ) const{
  return iValue;
}
void 
cond::FrontierProxy::setupSession(){
}
void 
cond::FrontierProxy::prepareConnection(){
}
void 
cond::FrontierProxy::prepareTransaction(){
}
DEFINE_EDM_PLUGIN(cond::TechnologyProxyFactory,cond::FrontierProxy,"frontier");
