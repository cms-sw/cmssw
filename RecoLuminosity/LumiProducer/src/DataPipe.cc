#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
lumi::DataPipe::DataPipe( const std::string& dest ):m_dest(dest),m_source(""),m_authpath(""){
}
void lumi::DataPipe::setSource( const std::string& source ){
  m_source=source;
}
void lumi::DataPipe::setAuthPath( const std::string& authpath ){
  m_authpath=authpath;
}
