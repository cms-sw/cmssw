#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
//#include <iostream>
void
cond::ConnectionHandler::registerConnection(const std::string& name,
					    const std::string& con,
					    int connectionTimeoutInSec){
  m_registry.insert(std::make_pair<std::string,cond::Connection*>(name, new cond::Connection(con,connectionTimeoutInSec)));
}
void 
cond::ConnectionHandler::registerConnection(const std::string& userconnect,
					    cond::DBSession& session,
					    int connectionTimeOutInSec){
  std::string realconnect(userconnect);
  std::string protocol;
  std::size_t pos=userconnect.find_first_of(':');
  if( pos!=std::string::npos ){
    protocol=userconnect.substr(0,pos);
    std::size_t p=protocol.find_first_of('_');
    if(p!=std::string::npos){
      protocol=protocol.substr(0,p);
    }
  }else{
    throw cond::Exception("connection string format error");
  }
  //std::cout<<"userconnect "<<userconnect<<std::endl;
  //std::cout<<"protocol "<<protocol<<std::endl;  
  std::auto_ptr<cond::TechnologyProxy> ptr(cond::TechnologyProxyFactory::get()->create(protocol,userconnect));
  realconnect=ptr->getRealConnectString();
  //std::cout<<"realconnect "<<realconnect<<std::endl;
  m_registry.insert(std::make_pair<std::string,cond::Connection*>(userconnect, new cond::Connection(realconnect,connectionTimeOutInSec)));
  ptr->setupSession(session);
}
void
cond::ConnectionHandler::removeConnection(const std::string& name ){
  std::map<std::string,cond::Connection*>::iterator pos=m_registry.find(name);
  if( pos!=m_registry.end() ){
    m_registry.erase(pos);
  }
}
cond::ConnectionHandler& 
cond::ConnectionHandler::Instance(){
  //std::cout<<"in ConnectionHandler::Instance"<<std::endl;
  if(!edmplugin::PluginManager::isAvailable()){
    edmplugin::PluginManager::Config config;
    const char* path = getenv("LD_LIBRARY_PATH");
    std::string spath(path? path: "");
    std::string::size_type last=0;
    std::string::size_type i=0;
    std::vector<std::string> paths;
    while( (i=spath.find_first_of(':',last))!=std::string::npos) {
      paths.push_back(spath.substr(last,i-last));
      last = i+1;
      //std::cout <<paths.back()<<std::endl;
    }
    paths.push_back(spath.substr(last,std::string::npos));
    config.searchPath(paths);
    edmplugin::PluginManager::configure(config);
  }
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
  if( m_registry.size()==0 ) return;
  std::map<std::string,cond::Connection*>::iterator it;  
  std::map<std::string,cond::Connection*>::iterator itEnd=m_registry.end();
  for(it=m_registry.begin();it!=itEnd;++it){
    if(it->second!=0){
      delete it->second;
      it->second=0;
    }
  }
  m_registry.clear();
}
cond::ConnectionHandler::~ConnectionHandler(){
  if( m_registry.size() != 0){
    this->disconnectAll();
  }
}
cond::Connection* 
cond::ConnectionHandler::getConnection( const std::string& name ){
  std::map<std::string,cond::Connection*>::iterator it=m_registry.find(name);
  if( it!=m_registry.end() ){
    return it->second;
  }
  return 0;
}
