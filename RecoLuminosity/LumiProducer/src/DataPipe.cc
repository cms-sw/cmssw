#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
lumi::DataPipe::DataPipe( const std::string& dest ):m_dest(dest),m_source(""),m_authpath(""),m_novalidate(false),m_norm(6370.0),m_nocheckingstablebeam(false){
}
void lumi::DataPipe::setNoValidate(){
  m_novalidate=true;
}
void lumi::DataPipe::setSource( const std::string& source ){
  m_source=source;
}
void lumi::DataPipe::setAuthPath( const std::string& authpath ){
  m_authpath=authpath;
}
void lumi::DataPipe::setMode( const std::string& mode ){
  m_mode=mode;
}
void lumi::DataPipe::setNorm( float norm ){
  m_norm=norm;
}
void lumi::DataPipe::setNoCheckingStableBeam(){
  m_nocheckingstablebeam=true;
}
std::string lumi::DataPipe::getSource()const{
  return m_source;
}
std::string lumi::DataPipe::getMode()const{
  return m_mode;
}
std::string lumi::DataPipe::getAuthPath()const{
  return m_authpath;
}
float lumi::DataPipe::getNorm()const{
  return m_norm;
}
