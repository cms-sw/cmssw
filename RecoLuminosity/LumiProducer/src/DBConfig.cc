#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "CoralKernel/Context.h"
#include "CoralKernel/IProperty.h"
#include "CoralKernel/IPropertyManager.h"
#include <boost/filesystem/operations.hpp>
namespace lumi{
  const std::string defaultAuthFileName="authentication.xml";
}
lumi::DBConfig::DBConfig(coral::ConnectionService& svc): m_svc(&svc){
}
lumi::DBConfig::~DBConfig(){
}
void lumi::DBConfig::setAuthentication( const std::string& authPath ){
  boost::filesystem::path boostAuthPath(authPath);
  if(boost::filesystem::is_directory(boostAuthPath)){
    boostAuthPath /= boost::filesystem::path(lumi::defaultAuthFileName);
  }
  std::string authFileName=boostAuthPath.string();
  coral::Context::instance().PropertyManager().property("AuthenticationFile")->set(authFileName);
  coral::Context::instance().loadComponent("CORAL/Services/XMLAuthenticationService");
}
std::string lumi::DBConfig::trueConnectStr( const std::string& usercon ){
  //empty for now
  return usercon;
}

