#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "PoolConnectionProxy.h"
#include "CoralConnectionProxy.h"
//#include <iostream>
cond::Connection::Connection(const std::string& con,
			     int connectionTimeOut):
  m_con( con ),
  m_connectionTimeOut ( connectionTimeOut )
{
  m_coralConnectionPool.reserve(10);
}
cond::Connection::~Connection(){
  this->disconnect();
}
void cond::Connection::connect( cond::DBSession* session ){
  m_connectionServiceHandle=&(session->connectionService());
  m_idleConnectionCleanupPeriod=session->configuration().connectionConfiguration()->idleConnectionCleanupPeriod();
}

/**if first time, init everything, start timer
   if not first time, 
*/
cond::CoralTransaction&
cond::Connection::coralTransaction(){
  if( !m_coralConnectionPool.empty() ){
    std::vector<cond::CoralConnectionProxy*>::iterator it;
    std::vector<cond::CoralConnectionProxy*>::iterator itEnd=m_coralConnectionPool.end();
    for( it=m_coralConnectionPool.begin(); it!=itEnd; ++it ){
      return static_cast<cond::CoralTransaction&>((*it)->transaction());
    }
  }
  cond::CoralConnectionProxy* me=new cond::CoralConnectionProxy(
	     m_connectionServiceHandle,m_con,m_connectionTimeOut,m_idleConnectionCleanupPeriod);
  m_coralConnectionPool.push_back(me);
  return static_cast<cond::CoralTransaction&>( me->transaction() );
}
//return transaction object(poolproxy, current transactionCounter, current time
cond::PoolTransaction&
cond::Connection::poolTransaction(){
  if( !m_poolConnectionPool.empty() ){
    std::vector<cond::PoolConnectionProxy*>::iterator it;
    std::vector<cond::PoolConnectionProxy*>::iterator itEnd=m_poolConnectionPool.end();
    for( it=m_poolConnectionPool.begin(); it!=itEnd; ++it ){
      return static_cast<cond::PoolTransaction&>((*it)->transaction());
    }
  }
  cond::PoolConnectionProxy* me=new cond::PoolConnectionProxy(m_connectionServiceHandle,m_con,m_connectionTimeOut,m_idleConnectionCleanupPeriod); 
  m_poolConnectionPool.push_back(me);
  return static_cast<cond::PoolTransaction&>(me->transaction());
}
std::string 
cond::Connection::connectStr() const{
  return m_con;
}
void
cond::Connection::disconnect(){
  if(!m_coralConnectionPool.empty()){
    std::vector<cond::CoralConnectionProxy*>::iterator it;
    std::vector<cond::CoralConnectionProxy*>::iterator itEnd=m_coralConnectionPool.end();
    for( it=m_coralConnectionPool.begin(); it!=itEnd; ++it){
      if( *it!=0 ) {
	delete *it;
	*it=0;
      } 
    }
    m_coralConnectionPool.clear();
  }
  if(!m_poolConnectionPool.empty()){
    std::vector<cond::PoolConnectionProxy*>::iterator poolit;
    std::vector<cond::PoolConnectionProxy*>::iterator poolitEnd=m_poolConnectionPool.end();
    for( poolit=m_poolConnectionPool.begin(); poolit!=poolitEnd; ++poolit){
      if( *poolit!=0 ) {
	delete *poolit;
	*poolit=0;
      } 
    }
    m_poolConnectionPool.clear();
  }
}
