#include "CondCore/DBCommon/interface/ConfigSessionFromParameterSet.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/filesystem/operations.hpp"
cond::ConfigSessionFromParameterSet::ConfigSessionFromParameterSet(
			cond::DBSession& session,
			const edm::ParameterSet& connectionPset ){
  std::string authDir=connectionPset.getUntrackedParameter<std::string>("authenticationPath","");
  boost::filesystem::path fs(authDir);
  int messageLevel=connectionPset.getUntrackedParameter<int>("messageLevel",0);
  bool enableConnectionSharing=connectionPset.getUntrackedParameter<bool>("enableConnectionSharing",true);
  int connectionTimeOut=connectionPset.getUntrackedParameter<int>("connectionTimeOut",600);
  bool enableReadOnlySessionOnUpdateConnection=connectionPset.getUntrackedParameter<bool>("enableReadOnlySessionOnUpdateConnection",true);
  bool loadBlobStreamer=connectionPset.getUntrackedParameter<bool>("loadBlobStreamer",false);
  int connectionRetrialPeriod=connectionPset.getUntrackedParameter<int>("connectionRetrialPeriod",30);
  int connectionRetrialTimeOut=connectionPset.getUntrackedParameter<int>("connectionRetrialTimeOut",180);
  if( fs.string().empty() ){
    session.sessionConfiguration().setAuthenticationMethod(cond::Env);
  }else{
    std::string authpath("CORAL_AUTH_PATH=");
    authpath+=fs.string();
    ::putenv(const_cast<char*>(authpath.c_str()));
    session.sessionConfiguration().setAuthenticationMethod(cond::XML);
  }  
  switch (messageLevel) {
  case 0 :
    session.sessionConfiguration().setMessageLevel( cond::Error );
    break;    
  case 1:
    session.sessionConfiguration().setMessageLevel( cond::Warning );
    break;
  case 2:
    session.sessionConfiguration().setMessageLevel( cond::Info );
    break;
  case 3:
    session.sessionConfiguration().setMessageLevel( cond::Debug );
    break;  
  default:
    session.sessionConfiguration().setMessageLevel( cond::Error );
  }
  if(enableConnectionSharing){
    session.connectionConfiguration().enableConnectionSharing();
  }
  session.connectionConfiguration().setConnectionTimeOut(connectionTimeOut);
  if(enableReadOnlySessionOnUpdateConnection){
    session.connectionConfiguration().enableReadOnlySessionOnUpdateConnections();
  }
  if(loadBlobStreamer){
    session.sessionConfiguration().setBlobStreamer("");
  }
  session.connectionConfiguration().setConnectionRetrialPeriod(connectionRetrialPeriod);
  session.connectionConfiguration().setConnectionRetrialTimeOut(connectionRetrialTimeOut);
}
