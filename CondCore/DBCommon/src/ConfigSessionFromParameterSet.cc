#include "CondCore/DBCommon/interface/ConfigSessionFromParameterSet.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "boost/filesystem/operations.hpp"
//#include <iostream>
cond::ConfigSessionFromParameterSet::ConfigSessionFromParameterSet(
		       cond::DBSession& session,
		       const edm::ParameterSet& connectionPset ){
  std::string xmlauthName=connectionPset.getUntrackedParameter<std::string>("authenticationPath","");
  int messageLevel=connectionPset.getUntrackedParameter<int>("messageLevel",0);
  bool enableConnectionSharing=connectionPset.getUntrackedParameter<bool>("enableConnectionSharing",true);
  int connectionTimeOut=connectionPset.getUntrackedParameter<int>("connectionTimeOut",600);
  bool enableReadOnlySessionOnUpdateConnection=connectionPset.getUntrackedParameter<bool>("enableReadOnlySessionOnUpdateConnection",true);
  bool loadBlobStreamer=connectionPset.getUntrackedParameter<bool>("loadBlobStreamer",false);
  int connectionRetrialPeriod=connectionPset.getUntrackedParameter<int>("connectionRetrialPeriod",30);
  int connectionRetrialTimeOut=connectionPset.getUntrackedParameter<int>("connectionRetrialTimeOut",180);
  bool enablePoolAutomaticCleanUp=connectionPset.getUntrackedParameter<bool>("enablePoolAutomaticCleanUp",false);
  if( xmlauthName.empty() ){
    session.configuration().setAuthenticationMethod(cond::Env);
  }else{
    session.configuration().setAuthenticationMethod(cond::XML);
    session.configuration().setAuthenticationPath(xmlauthName);
  }  
  switch (messageLevel) {
  case 0 :
    session.configuration().setMessageLevel( cond::Error );
    break;    
  case 1:
    session.configuration().setMessageLevel( cond::Warning );
    break;
  case 2:
    session.configuration().setMessageLevel( cond::Info );
    break;
  case 3:
    session.configuration().setMessageLevel( cond::Debug );
    break;  
  default:
    session.configuration().setMessageLevel( cond::Error );
  }
  if(enableConnectionSharing){
    session.configuration().connectionConfiguration()->enableConnectionSharing();
  }else{
    session.configuration().connectionConfiguration()->disableConnectionSharing();
  }
  session.configuration().connectionConfiguration()->setConnectionTimeOut(connectionTimeOut);
  if(enableReadOnlySessionOnUpdateConnection){
    session.configuration().connectionConfiguration()->enableReadOnlySessionOnUpdateConnections();
  }else{
    session.configuration().connectionConfiguration()->disableReadOnlySessionOnUpdateConnections();
  }
  if( enablePoolAutomaticCleanUp ){
    session.configuration().connectionConfiguration()->enablePoolAutomaticCleanUp();
  }else{
    session.configuration().connectionConfiguration()->disablePoolAutomaticCleanUp();
  }
  if(loadBlobStreamer){
    session.configuration().setBlobStreamer("");
  }
  session.configuration().connectionConfiguration()->setConnectionRetrialPeriod(connectionRetrialPeriod);
  session.configuration().connectionConfiguration()->setConnectionRetrialTimeOut(connectionRetrialTimeOut);
}
