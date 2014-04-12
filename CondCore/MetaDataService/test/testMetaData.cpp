#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"

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

    //test errors
    coralDb.transaction().start(false);
    if(coralDb.storage().exists()){
      coralDb.storage().drop();
    }
    coralDb.createDatabase();

    coralDb.transaction().commit();

    coralDb.transaction().start(true);
    cond::MetaData metadata(coralDb);
    if(metadata.hasTag("crap")) std::cout << "ERROR: wrong assertion" << std::endl;
    /**
    try {
      cond::MetaDataEntry result; 
      metadata.getEntryByTag("crap", result);
    } catch (cond::Exception const & ce) {
      std::cout << "OK " << ce.what() << std::endl;
    }
    **/
    try { 
      metadata.getToken("crap");
    } catch (cond::Exception const & ce) {
      std::cout << "OK " << ce.what() << std::endl;
    }
    try { 
      std::vector<std::string> alltags;
      metadata.listAllTags(alltags);
    } catch (cond::Exception const & ce) {
      std::cout << "OK " << ce.what() << std::endl;
    }

    coralDb.transaction().commit();


    coralDb.transaction().start(false);
    std::string t1("token1");
    metadata.addMapping("mytest1",t1);
    coralDb.transaction().commit();

    //test errors
    coralDb.transaction().start(true);
    if(metadata.hasTag("crap")) std::cout << "wrong: crap shall not be there" << std::endl;
    /**
    try {
      cond::MetaDataEntry result; 
      metadata.getEntryByTag("crap", result);
    } catch (cond::Exception const & ce) {
      std::cout << "OK " << ce.what() << std::endl;
    }
    **/
    try { 
      metadata.getToken("crap");
    } catch (cond::Exception const & ce) {
      std::cout << "OK " << ce.what() << std::endl;
    }
    coralDb.transaction().commit();



    std::string t2("token2");
    coralDb.transaction().start(false);
    metadata.addMapping("mytest2",t2,cond::timestamp);
    coralDb.transaction().commit();

    coralDb.transaction().start(true);
    if(!metadata.hasTag("mytest1")) std::cout << "wrong: mytest1 IS there" << std::endl;
    std::string tok1=metadata.getToken("mytest1");
    cond::MetaDataEntry r1;
    /**
    metadata.getEntryByTag("mytest1",r1);
    **/
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
    std::string tok2=metadata.getToken("mytest2");
    cond::MetaDataEntry r2;
    /**
    metadata.getEntryByTag("mytest2",r2);
    **/
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
    /**
    coralDb.transaction().start(false);
     metadata.replaceToken("mytest2",newtok2);
    coralDb.transaction().commit();
    **/

    coralDb.transaction().start(true);
    std::string mytok2=metadata.getToken("mytest2");
    std::cout<<"get back new tok2 "<<newtok2<<" "<<mytok2<<std::endl;
    std::cout<<"tag exists mytest2 "<<metadata.hasTag("mytest2")<<std::endl;
    std::cout<<"tag exists crap "<<metadata.hasTag("crap")<<std::endl;
    std::vector<std::string> alltags;
    metadata.listAllTags(alltags);
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


