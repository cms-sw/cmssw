#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
cond::ConnectionConfiguration::ConnectionConfiguration():m_enableConSharing(false),m_connectionRetrialPeriod(0),m_connectionRetrialTimeOut(0),m_connectionTimeOut(0),m_enableCommonConnection(false){}
cond::ConnectionConfiguration::~ConnectionConfiguration(){}
void cond::ConnectionConfiguration::enableConnectionSharing(){
  m_enableConSharing=true;
}
bool cond::ConnectionConfiguration::isConnectionSharingEnabled() const{
  return m_enableConSharing;
}
void cond::ConnectionConfiguration::setConnectionRetrialPeriod( int timeInSeconds ){
  m_connectionRetrialPeriod=timeInSeconds;
}
int cond::ConnectionConfiguration::connectionRetrialPeriod() const{
  return m_connectionRetrialPeriod;
}
void cond::ConnectionConfiguration::setConnectionRetrialTimeOut( int timeOutInSeconds ){
  m_connectionRetrialTimeOut=timeOutInSeconds;
}
int cond::ConnectionConfiguration::connectionRetrialTimeOut() const{
  return m_connectionRetrialTimeOut;
}
void cond::ConnectionConfiguration::setConnectionTimeOut( int timeOutInSeconds ){
  m_connectionTimeOut=timeOutInSeconds;
}
int cond::ConnectionConfiguration::connectionTimeOut(){
  return m_connectionTimeOut;
}
void cond::ConnectionConfiguration::enableReadOnlySessionOnUpdateConnections(){
  m_enableCommonConnection=true;
}
void cond::ConnectionConfiguration::disableReadOnlySessionOnUpdateConnections(){
  m_enableCommonConnection=false;
}
bool cond::ConnectionConfiguration::isReadOnlySessionOnUpdateConnectionsEnabled(){
  return m_enableCommonConnection;
}
