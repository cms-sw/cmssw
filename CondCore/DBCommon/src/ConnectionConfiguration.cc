#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
//#include <iostream>
cond::ConnectionConfiguration::ConnectionConfiguration():m_enableConSharing(true),m_connectionRetrialPeriod(10),m_connectionRetrialTimeOut(60),m_connectionTimeOut(300),m_idleconnectionCleanupPeriod(10),m_enableCommonConnection(false),m_enablePoolAutomaticCleanUp(true),m_monitorLevel(coral::monitor::Off){}
cond::ConnectionConfiguration::~ConnectionConfiguration(){}
void cond::ConnectionConfiguration::enableConnectionSharing(){
  m_enableConSharing=true;
}
void cond::ConnectionConfiguration::disableConnectionSharing(){
  m_enableConSharing=false;
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
void cond::ConnectionConfiguration::setIdleConnectionCleanupPeriod( int timeInSeconds ){
  m_idleconnectionCleanupPeriod=timeInSeconds;
}
int cond::ConnectionConfiguration::idleConnectionCleanupPeriod() const{
  return m_idleconnectionCleanupPeriod;
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
void cond::ConnectionConfiguration::enablePoolAutomaticCleanUp(){
  m_enablePoolAutomaticCleanUp=true;
}
void cond::ConnectionConfiguration::disablePoolAutomaticCleanUp(){
  m_enablePoolAutomaticCleanUp=false;
}
bool cond::ConnectionConfiguration::isPoolAutomaticCleanUpEnabled() const{
  return m_enablePoolAutomaticCleanUp;
}
coral::monitor::Level 
cond::ConnectionConfiguration::monitorLevel() const{
  return m_monitorLevel;
}
