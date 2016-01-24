#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"
#include "CondCore/DBCommon/interface/Logger.h"
#include "CondCore/ORA/interface/PoolToken.h"
#include "CondCore/ORA/interface/SharedLibraryName.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/PluginManager/interface/standard.h"
#include <string>
#include <iostream>
//#include <stdio.h>
//#include <time.h>
#include <unistd.h>

namespace cond{
  class TokenBuilder{
  public:
    TokenBuilder(): m_token("") {
    }

    ~TokenBuilder() {
    }

    void set( const std::string& dictLib,
	      const std::string& className,
	      const std::string& containerName,
	      int pkcolumnValue=0) {
      
      ora::SharedLibraryName libName;
      edmplugin::SharedLibrary shared( libName(dictLib) );
      m_token = writeToken(containerName, 0, pkcolumnValue, className);
    }

    std::string const & tokenAsString() const {
      return m_token;
    }
  private:
    std::string m_token;
  };
}//ns cond

int main(){
  cond::TokenBuilder tk;
  tk.set("CondFormatsCalibration",
	 "Pedestals",
	 "Pedestals",
	 0);
  std::string const tok1 = tk.tokenAsString();
  tk.set("CondFormatsCalibration",
	 "Pedestals",
	 "Pedestals",
	 1);
  std::string const tok2 = tk.tokenAsString();
  std::string constr("sqlite_file:unittest_DBLogger.db");
  //std::string constr("oracle://devdb10/cms_xiezhen_dev");
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  cond::DbConnection connection;
  connection.configuration().setMessageLevel( coral::Error );
  connection.configure();
  cond::DbSession session = connection.createSession();
  session.open( constr );
  cond::Logger mylogger( session );
  cond::UserLogInfo a;
  a.provenance="me";
  mylogger.createLogDBIfNonExist();
  mylogger.logOperationNow(a,constr,std::string("Pedestals"),tok1,"mytag1","runnumber",0,1);
  std::cout<<"1. waiting"<<std::endl;
  sleep(1);
  std::cout<<"1. stop waiting"<<std::endl;
  std::cout<<"1. waiting"<<std::endl;
  sleep(1);
  std::cout<<"1. stop waiting"<<std::endl;
  mylogger.logFailedOperationNow(a,constr,std::string("Pedestals"),tok1,"mytag1","runnumber",1,1,"EOOROR");
  std::cout<<"1. waiting"<<std::endl;
  sleep(1);
  std::cout<<"1. stop waiting"<<std::endl;
  
  std::cout<<"1. waiting"<<std::endl;
  sleep(1);
  std::cout<<"1. stop waiting"<<std::endl;
  mylogger.logOperationNow(a,constr,std::string("Pedestals"),tok2,"mytag","runnumber",1,2);
  std::cout<<"1. waiting"<<std::endl;
  sleep(1);
  std::cout<<"1. stop waiting"<<std::endl;
  /*std::cout<<"about to lookup last entry"<<std::endl;
  cond::LogDBEntry result;
  mylogger.LookupLastEntryByProvenance("me",result);
  std::cout<<"result \n";
  std::cout<<"logId "<<result.logId<<"\n";
  std::cout<<"destinationDB "<<result.destinationDB<<"\n";
  std::cout<<"provenance "<<result.provenance<<"\n";
  std::cout<<"usertext "<<result.usertext<<"\n";
  std::cout<<"iovtag "<<result.iovtag<<"\n";
  std::cout<<"iovtimetype "<<result.iovtimetype<<"\n";
  std::cout<<"payloadIdx "<<result.payloadIdx<<"\n";
  std::cout<<"payloadName "<<result.payloadName<<"\n";
  std::cout<<"payloadToken "<<result.payloadToken<<"\n";
  std::cout<<"payloadContainer "<<result.payloadContainer<<"\n";
  std::cout<<"exectime "<<result.exectime<<"\n";
  std::cout<<"execmessage "<<result.execmessage<<std::endl;
  cond::LogDBEntry result2;
  mylogger.LookupLastEntryByTag("mytag1",result2);
  std::cout<<"result2 \n";
  std::cout<<"logId "<<result2.logId<<"\n";
  std::cout<<"destinationDB "<<result2.destinationDB<<"\n";
  std::cout<<"provenance "<<result2.provenance<<"\n";
  std::cout<<"usertext "<<result2.usertext<<"\n";
  std::cout<<"iovtag "<<result2.iovtag<<"\n";
  std::cout<<"iovtimetype "<<result2.iovtimetype<<"\n";
  std::cout<<"payloadIdx "<<result2.payloadIdx<<"\n";
  std::cout<<"payloadName "<<result2.payloadName<<"\n";
  std::cout<<"payloadToken "<<result2.payloadToken<<"\n";
  std::cout<<"payloadContainer "<<result2.payloadContainer<<"\n";
  std::cout<<"exectime "<<result2.exectime<<"\n";
  std::cout<<"execmessage "<<result2.execmessage<<std::endl;
  */
  //coralTransaction.commit();
}
