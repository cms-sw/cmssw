#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/DBSession.h"
void 
cond::ConnectionHandler::registerConnection(const std::string& name,
					    const std::string& con,
					    const std::string& filecatalog,
					    unsigned int connectionTimeout){
  m_registry.insert(std::make_pair<std::string,cond::Connection*>(name,new cond::Connection(con,filecatalog,connectionTimeout)));
}
void
cond::ConnectionHandler::registerConnection(const std::string& name,
					    const std::string& con,
					    unsigned int connectionTimeout){
  m_registry.insert(std::make_pair<std::string,cond::Connection*>(name, new cond::Connection(con,connectionTimeout)));
}
cond::ConnectionHandler& 
cond::ConnectionHandler::Instance(){
  static cond::ConnectionHandler me;
  return me;
}
void
cond::ConnectionHandler::connect(cond::DBSession* session){
  std::map<std::string,cond::Connection*>::iterator it;  
  std::map<std::string,cond::Connection*>::iterator itEnd=m_registry.end();
  for(it=m_registry.begin();it!=itEnd;++it){
    it->second->connect(session);
  }
}
void 
cond::ConnectionHandler::disconnectAll(){
}
cond::ConnectionHandler::~ConnectionHandler(){
  std::map<std::string,cond::Connection*>::iterator it;  
  std::map<std::string,cond::Connection*>::iterator itEnd=m_registry.end();
  for(it=m_registry.begin();it!=itEnd;++it){
    delete it->second;
    it->second=0;
  }
  m_registry.clear();
}
cond::Connection* 
cond::ConnectionHandler::getConnection( const std::string& name ){
  std::map<std::string,cond::Connection*>::iterator it=m_registry.find(name);
  if( it!=m_registry.end() ){
    return it->second;
  }
  return 0;
}
