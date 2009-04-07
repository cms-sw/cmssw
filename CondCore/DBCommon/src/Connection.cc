#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "PoolConnectionProxy.h"
#include "CoralConnectionProxy.h"

//#include <iostream>
cond::Connection::Connection(const std::string& con,
			     int connectionTimeOut):
  m_con( con ),
  m_connectionTimeOut ( connectionTimeOut ),
  m_connectionServiceHandle(0),
  m_blobstreamingServiceHandle(0)
{}

cond::Connection::~Connection(){
  this->disconnect();
}
void cond::Connection::connect( cond::DBSession* session ){
  m_connectionServiceHandle=&(session->connectionService());
  m_blobstreamingServiceHandle=&(session->blobStreamingService());
  m_idleConnectionCleanupPeriod=session->configuration().connectionConfiguration()->idleConnectionCleanupPeriod();
}

/**if first time, init everything, start timer
   if not first time, 
*/
cond::CoralTransaction&
cond::Connection::coralTransaction(){
  if (!m_coralConnection.get())  
    m_coralConnection.reset(new cond::CoralConnectionProxy(m_connectionServiceHandle,m_con,m_connectionTimeOut,m_idleConnectionCleanupPeriod));

  return static_cast<cond::CoralTransaction&>(m_coralConnection->transaction());

}
//return transaction object(poolproxy, current transactionCounter, current time
cond::PoolTransaction&
cond::Connection::poolTransaction(){

  if(!m_poolConnection.get())
    m_poolConnection.reset(new cond::PoolConnectionProxy(m_connectionServiceHandle,m_blobstreamingServiceHandle,m_con,m_connectionTimeOut,m_idleConnectionCleanupPeriod)); 
 

  return static_cast<cond::PoolTransaction&>(m_poolConnection->transaction());
}

std::string 
cond::Connection::connectStr() const{
  return m_con;
}

void
cond::Connection::disconnect(){
    m_coralConnection.reset();
    m_poolConnection.reset();
}

