#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/ConnectionPool.h"
// externals
#include "RelationalAccess/ISessionProxy.h"

ora::SharedSession::SharedSession():
  m_proxy(){
}

ora::SharedSession::SharedSession(boost::shared_ptr<coral::ISessionProxy>& coralSession):
  m_proxy(coralSession){
}

ora::SharedSession::SharedSession( const ora::SharedSession& rhs ):
  m_proxy( rhs.m_proxy ){
}

ora::SharedSession::~SharedSession(){
}

ora::SharedSession& ora::SharedSession::operator=( const ora::SharedSession& rhs ){
  m_proxy = rhs.m_proxy;
  return *this;
}

coral::ISessionProxy&
ora::SharedSession::get(){
  if(!m_proxy.get()){
    throwException("Coral Database Session is not available.",
                   "SharedSession::proxy");
  }
  return *m_proxy;
}

bool ora::SharedSession::isValid(){
  return m_proxy.get();
}

void ora::SharedSession::close(){
  m_proxy.reset();
}

ora::ConnectionPool::ConnectionPool():m_connectionService(),m_sessions(){
}

ora::ConnectionPool::~ConnectionPool(){
}

coral::IConnectionService& ora::ConnectionPool::connectionService(){
  return m_connectionService;
}
coral::IConnectionServiceConfiguration& ora::ConnectionPool::configuration(){
  return m_connectionService.configuration();
}

ora::SharedSession ora::ConnectionPool::connect( const std::string& connectionString,
						 coral::AccessMode accessMode ){
  bool valid = false;
  boost::shared_ptr<coral::ISessionProxy> session;
  std::map<std::string,boost::weak_ptr<coral::ISessionProxy> >::iterator iS = m_sessions.find( lookupString( connectionString, accessMode ) );
  if( iS != m_sessions.end() ){
    if( !iS->second.expired() ){
      session = iS->second.lock();
      valid = true;
    } 
  } else {
    iS = m_sessions.insert(std::make_pair( lookupString( connectionString, accessMode ),boost::weak_ptr<coral::ISessionProxy>())).first;
  }
  if(!valid){
    session.reset(m_connectionService.connect( connectionString, accessMode ));
    boost::weak_ptr<coral::ISessionProxy> tmp(session);
    iS->second.swap( tmp );
  }
  return SharedSession( session );
}

ora::SharedSession ora::ConnectionPool::connect( const std::string& connectionString,
						 const std::string& asRole,
						 coral::AccessMode accessMode ){
  bool valid = false;
  boost::shared_ptr<coral::ISessionProxy> session;
  std::map<std::string,boost::weak_ptr<coral::ISessionProxy> >::iterator iS = m_sessions.find( lookupString( connectionString, asRole, accessMode ) );
  if( iS != m_sessions.end() ){
    if( !iS->second.expired() ){
      session = iS->second.lock();
      valid = true;
    } 
  } else {
    iS = m_sessions.insert(std::make_pair( lookupString( connectionString, asRole, accessMode ),boost::weak_ptr<coral::ISessionProxy>())).first;
  }
  if(!valid){
    session.reset(m_connectionService.connect( connectionString, asRole, accessMode ));
    boost::weak_ptr<coral::ISessionProxy> tmp(session);
    iS->second.swap( tmp );
  }
  return SharedSession( session );
}


std::string ora::ConnectionPool::lookupString( const std::string& connectionString,
                                               coral::AccessMode accessMode ){
  std::stringstream lookupString;
  lookupString << "["<<connectionString << "]_";
  if(accessMode == coral::ReadOnly){
    lookupString << "R";
  } else {
    lookupString << "W";
  }
  return lookupString.str();
}

std::string ora::ConnectionPool::lookupString( const std::string& connectionString,
					       const std::string& role,
                                               coral::AccessMode accessMode ){
  std::string rolePrefix(role);
  return rolePrefix+lookupString( connectionString, accessMode );
}
