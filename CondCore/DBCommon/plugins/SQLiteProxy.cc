#include "SQLiteProxy.h"
#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
cond::SQLiteProxy::SQLiteProxy(){
}
cond::SQLiteProxy::~SQLiteProxy(){
}
std::string 
cond::SQLiteProxy::getRealConnectString( const std::string& iValue ) const{
  return iValue;
}
void 
cond::SQLiteProxy::setupSession(){
}
void 
cond::SQLiteProxy::prepareConnection(){
}
void 
cond::SQLiteProxy::prepareTransaction(){
}
DEFINE_EDM_PLUGIN(cond::TechnologyProxyFactory,cond::SQLiteProxy,"sqlite");
