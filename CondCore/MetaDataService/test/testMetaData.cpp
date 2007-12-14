#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "SealKernel/IMessageService.h"
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <iostream>
int main(){
  ::putenv("CORAL_AUTH_USER=cms_xiezhen_dev");
  ::putenv("CORAL_AUTH_PASSWORD=xiezhen123");
  try{
    cond::DBSession* session=new cond::DBSession(true);
    session->sessionConfiguration().setMessageLevel(cond::Debug);
    session->connectionConfiguration().setConnectionTimeOut(0);
    session->open();
    cond::RelationalStorageManager coraldb("sqlite_file:pippo.db",session);
    cond::MetaData metadata_svc(coraldb);
    std::string t1("token1");
    coraldb.connect(cond::ReadWriteCreate);
    coraldb.startTransaction(false);
    metadata_svc.addMapping("mytest1",t1);
    coraldb.commit();
    coraldb.disconnect();
    std::string t2("token2");
    coraldb.connect(cond::ReadWriteCreate);
    coraldb.startTransaction(false);
    metadata_svc.addMapping("mytest2",t2);
    coraldb.commit();
    coraldb.disconnect();
    std::cout<<"clean up idle connections"<<std::endl;
    //session->purgeConnections();
    coraldb.connect(cond::ReadOnly);
    coraldb.startTransaction(true);
    std::string tok1=metadata_svc.getToken("mytest2");
    coraldb.commit();
    coraldb.disconnect();
    std::cout<<"got token1 "<<tok1<<std::endl;
    coraldb.connect(cond::ReadOnly);
    coraldb.startTransaction(true);
    std::string tok2=metadata_svc.getToken("mytest2");
    coraldb.commit();
    coraldb.disconnect();
    std::cout<<"got token2 "<<tok2<<std::endl;
    std::string newtok2="newtoken2";
    coraldb.connect(cond::ReadWriteCreate);
    coraldb.startTransaction(false);
    metadata_svc.replaceToken("mytest2",newtok2);
    coraldb.commit();
    coraldb.disconnect();
    coraldb.connect(cond::ReadOnly);
    coraldb.startTransaction(true);
    std::string mytok2=metadata_svc.getToken("mytest2");
    std::cout<<"get back new tok2 "<<newtok2<<" "<<mytok2<<std::endl;
    std::cout<<"tag exists mytest2 "<<metadata_svc.hasTag("mytest2")<<std::endl;
    std::cout<<"tag exists crap "<<metadata_svc.hasTag("crap")<<std::endl;
    std::vector<std::string> alltags;
    metadata_svc.listAllTags(alltags);
    coraldb.commit();
    coraldb.disconnect();
    std::copy (alltags.begin(),
	       alltags.end(),
	       std::ostream_iterator<std::string>(std::cout,"\n")
	       );
    //metadata_svc.deleteAllEntries();
    //metadata_svc.disconnect();
    session->close();
    delete session;
  }catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(std::exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Funny error"<<std::endl;
  }
}


