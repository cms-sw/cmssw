#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "PoolConnectionProxy.h"
#include "CoralConnectionProxy.h"
cond::Connection::Connection(const std::string& con,
			     const std::string& catalog,
			     unsigned int connectionTimeOut):
  m_con( con ),
  m_catalog( catalog ),
  m_connectionTimeOut ( connectionTimeOut )
{
  m_poolConnectionPool.reserve(10);
  m_coralConnectionPool.reserve(10);
}
cond::Connection::Connection(const std::string& con,
			     unsigned int connectionTimeOut):
  m_con( con ),
  m_catalog( "" ),
  m_connectionTimeOut ( connectionTimeOut )
{
  m_coralConnectionPool.reserve(10);
}
cond::Connection::~Connection(){
  std::vector<cond::CoralConnectionProxy*>::iterator it;
  std::vector<cond::CoralConnectionProxy*>::iterator itEnd=m_coralConnectionPool.end();
  for( it=m_coralConnectionPool.begin(); it!=itEnd; ++it){
    if( *it!=0 ) {
      delete *it;
      *it=0;
    } 
  }
  std::vector<cond::PoolConnectionProxy*>::iterator poolit;
  std::vector<cond::PoolConnectionProxy*>::iterator poolitEnd=m_poolConnectionPool.end();
  for( poolit=m_poolConnectionPool.begin(); poolit!=poolitEnd; ++poolit){
    if( *poolit!=0 ) {
      delete *poolit;
      *poolit=0;
    } 
  }
}
void cond::Connection::connect( cond::DBSession* session ){
  m_connectionServiceHandle=&(session->connectionService());
}
/**if first time, init everything, start timer
   if not first time, 
*/
cond::CoralTransaction&
cond::Connection::coralTransaction(bool isReadOnly){
  if( m_coralConnectionPool.size()==0 ){
    cond::CoralConnectionProxy* me=new cond::CoralConnectionProxy(
    m_connectionServiceHandle,m_con,isReadOnly,m_connectionTimeOut); 
    m_coralConnectionPool.push_back(me);
    return static_cast<cond::CoralTransaction&>( me->transaction() );
  }else{
    std::vector<cond::CoralConnectionProxy*>::iterator it;
    std::vector<cond::CoralConnectionProxy*>::iterator itEnd=m_coralConnectionPool.end();
    for( it=m_coralConnectionPool.begin(); it!=itEnd; ++it ){
      if( (*it)->isReadOnly() == isReadOnly){ //found same type
	return static_cast<cond::CoralTransaction&>((*it)->transaction());
      }else{
	cond::CoralConnectionProxy* me=new cond::CoralConnectionProxy(
           m_connectionServiceHandle,m_con,isReadOnly,m_connectionTimeOut); 
	m_coralConnectionPool.push_back(me);
	return static_cast<cond::CoralTransaction&>( me->transaction() );
      }
    }
  }
}
//return transaction object(poolproxy, current transactionCounter, current time
cond::PoolTransaction&
cond::Connection::poolTransaction(bool isReadOnly){
  if( m_poolConnectionPool.size()==0 ){
    cond::PoolConnectionProxy* me=new cond::PoolConnectionProxy(m_con,m_catalog,isReadOnly,m_connectionTimeOut); 
    m_poolConnectionPool.push_back(me);
    return static_cast<cond::PoolTransaction&>(me->transaction());
  }else{
    std::vector<cond::PoolConnectionProxy*>::iterator it;
    std::vector<cond::PoolConnectionProxy*>::iterator itEnd=m_poolConnectionPool.end();
    for( it=m_poolConnectionPool.begin(); it!=itEnd; ++it ){
      if( (*it)->isReadOnly() == isReadOnly){ //found same type
	return static_cast<cond::PoolTransaction&>((*it)->transaction());
      }else{
	 cond::PoolConnectionProxy* me=new cond::PoolConnectionProxy(m_con,m_catalog,isReadOnly,m_connectionTimeOut); 
	 m_poolConnectionPool.push_back(me);
	 return static_cast<cond::PoolTransaction&>(me->transaction());
      }
    }
  }
}
std::string 
cond::Connection::connectStr() const{
  return m_con;
}
std::string 
cond::Connection::catalogStr() const{
  return m_catalog;
}
