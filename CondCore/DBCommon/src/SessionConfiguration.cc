#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
cond::SessionConfiguration::SessionConfiguration():m_authMethod(cond::Env),m_hasBlobstreamer(false),m_blobstreamerName(""),m_messageLevel(cond::Error),m_conConfig(new ConnectionConfiguration){}
cond::SessionConfiguration::~SessionConfiguration(){
  delete m_conConfig;
}
cond::ConnectionConfiguration* 
cond::SessionConfiguration::connectionConfiguration(){
  return m_conConfig;
}
void 
cond::SessionConfiguration::setAuthenticationMethod( cond::AuthenticationMethod m ){
  m_authMethod=m;
}
void 
cond::SessionConfiguration::setAuthenticationPath( const std::string& p ){
  m_authPath=p;
}
void 
cond::SessionConfiguration::setBlobStreamer( const std::string& name ){
  m_hasBlobstreamer=true;
  m_blobstreamerName=name;
}
void 
cond::SessionConfiguration::setMessageLevel( cond::MessageLevel l ){
  m_messageLevel=l;
}
cond::AuthenticationMethod 
cond::SessionConfiguration::authenticationMethod() const{
  return m_authMethod;
}
bool 
cond::SessionConfiguration::hasBlobStreamService() const{
  return m_hasBlobstreamer;
}
std::string 
cond::SessionConfiguration::blobStreamerName() const{
  return m_blobstreamerName;
}
cond::MessageLevel 
cond::SessionConfiguration::messageLevel() const{
  return m_messageLevel;
}
std::string 
cond::SessionConfiguration::authName() const{
  return m_authPath;
}
