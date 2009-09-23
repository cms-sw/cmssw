#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"

#include "CondCore/MetaDataService/interface/MetaDataSchemaUtility.h"
#include "CondCore/DBCommon/interface/Time.h"
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <iostream>
int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  try{
    cond::DbConnection connection;
    connection.configuration().setPoolAutomaticCleanUp( false );
    connection.configure();
    cond::DbSession coralDb = connection.createSession();
    coralDb.open("sqlite_file:meta.db");
    cond::MetaData metadata_svc(coralDb);
    coralDb.transaction().start(false);
    std::string t1("token1");
    metadata_svc.addMapping("mytest1",t1);
    coralDb.transaction().commit();
    std::string t2("token2");
    coralDb.transaction().start(false);
    metadata_svc.addMapping("mytest2",t2,cond::timestamp);
    coralDb.transaction().commit();
    coralDb.transaction().start(true);
    std::string tok1=metadata_svc.getToken("mytest1");
    cond::MetaDataEntry r1;
    metadata_svc.getEntryByTag("mytest1",r1);
    coralDb.transaction().commit();
    std::cout<<"got token1 "<<tok1<<std::endl;
    std::cout<<"got entry tagname "<<r1.tagname<<std::endl;
    std::cout<<"got entry iovtoken "<<r1.iovtoken<<std::endl;
    if(r1.timetype==cond::runnumber){
      std::cout<<"runnumber"<<std::endl;
    }else{
      std::cout<<"timestamp"<<std::endl;
    }
    coralDb.transaction().start(true);
    std::string tok2=metadata_svc.getToken("mytest2");
    cond::MetaDataEntry r2;
    metadata_svc.getEntryByTag("mytest2",r2);
    coralDb.transaction().commit();
    std::cout<<"got token2 "<<tok2<<std::endl;
    std::cout<<"got entry tagname "<<r2.tagname<<std::endl;
    std::cout<<"got entry iovtoken "<<r2.iovtoken<<std::endl;
    if(r2.timetype==cond::runnumber){
      std::cout<<"runnumber"<<std::endl;
    }else{
      std::cout<<"timestamp"<<std::endl;
    }
    std::string newtok2="newtoken2";
    coralDb.transaction().start(false);
    //coralTransaction.start();
    metadata_svc.replaceToken("mytest2",newtok2);
    coralDb.transaction().commit();
    //coralTransaction.commit();
    //coralTransaction.start();
    coralDb.transaction().start(true);
    std::string mytok2=metadata_svc.getToken("mytest2");
    std::cout<<"get back new tok2 "<<newtok2<<" "<<mytok2<<std::endl;
    std::cout<<"tag exists mytest2 "<<metadata_svc.hasTag("mytest2")<<std::endl;
    std::cout<<"tag exists crap "<<metadata_svc.hasTag("crap")<<std::endl;
    std::vector<std::string> alltags;
    metadata_svc.listAllTags(alltags);
    coralDb.transaction().commit();
    coralDb.transaction().start(false);
    cond::MetaDataSchemaUtility ut(coralDb);
    ut.drop();
    ut.drop();
    coralDb.transaction().commit();
    std::copy (alltags.begin(),
               alltags.end(),
               std::ostream_iterator<std::string>(std::cout,"\n")
      );
  }catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(std::exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Funny error"<<std::endl;
  }
}


