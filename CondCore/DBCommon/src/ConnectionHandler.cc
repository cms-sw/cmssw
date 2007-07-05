#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/DBSession.h"
void 
cond::ConnectionHandler::registerConnection(const std::string& con,
					    const std::string& filecatalog,
					    unsigned int connectionTimeout){
  m_registry.push_back(new cond::Connection(con,filecatalog,connectionTimeout) );
}
void
cond::ConnectionHandler::registerConnection(const std::string& con,
					    unsigned int connectionTimeout){
  m_registry.push_back( new cond::Connection(con,connectionTimeout) );
}
cond::ConnectionHandler& 
cond::ConnectionHandler::Instance(){
  static cond::ConnectionHandler me;
  return me;
}
void
cond::ConnectionHandler::connect(cond::DBSession* session){
  std::vector<cond::Connection*>::iterator it;  
  std::vector<cond::Connection*>::iterator itEnd=m_registry.end();
  for(it=m_registry.begin();it!=itEnd;++it){
    (*it)->connect(session);
  }
}
//void
//cond::ConnectionHandler::disconnectAll(){
/* std::vector<cond::Connection*>::iterator it; 
   std::vector<cond::Connection*>::iterator itEnd=m_registry.end();
   for(it=m_registry.begin();it!=itEnd;++it){
   (*it)->disconnect();
   }
**/
//}
cond::ConnectionHandler::~ConnectionHandler(){
  std::vector<cond::Connection*>::iterator it;  
  std::vector<cond::Connection*>::iterator itEnd=m_registry.end();
  for(it=m_registry.begin();it!=itEnd;++it){
    delete *it;
    (*it)=0;
  }
  m_registry.clear();
}
