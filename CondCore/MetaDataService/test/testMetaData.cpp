#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <iostream>
int main(){
  ::putenv("CORAL_AUTH_USER=cms_xiezhen_dev");
  ::putenv("CORAL_AUTH_PASSWORD=xiezhen123");
  try{
    cond::DBSession* session=new cond::DBSession;
    session->configuration().setMessageLevel(cond::Error);
    session->configuration().setAuthenticationMethod(cond::XML);
    static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
    conHandler.registerConnection("mysqlite1","sqlite_file:pippo.db",0);
    session->open();
    conHandler.connect(session);
    std::cout<<1<<std::endl;
    cond::Connection* myconnection=conHandler.getConnection("mysqlite1");
    std::cout<<"myconnection "<<myconnection<<std::endl;
    cond::CoralTransaction& coralTransaction=myconnection->coralTransaction(false);
    std::cout<<2<<&coralTransaction<<std::endl;
    cond::MetaData metadata_svc(coralTransaction);
    std::cout<<3<<&coralTransaction<<std::endl;
    coralTransaction.start();
    std::cout<<4<<&coralTransaction<<std::endl;
    std::string t1("token1");
    metadata_svc.addMapping("mytest1",t1);
    coralTransaction.commit();
    std::cout<<6<<&coralTransaction<<std::endl;
    std::string t2("token2");
    std::cout<<6.5<<&coralTransaction<<std::endl;
    coralTransaction.start();
    std::cout<<7<<std::endl;
    metadata_svc.addMapping("mytest2",t2);
    std::cout<<8<<std::endl;
    coralTransaction.commit();
    std::cout<<9<<std::endl;
    coralTransaction.start();
    std::cout<<10<<std::endl;
    std::string tok1=metadata_svc.getToken("mytest2");
    std::cout<<11<<std::endl;
    coralTransaction.commit();
    std::cout<<12<<std::endl;
    std::cout<<"got token1 "<<tok1<<std::endl;
    //coralTransaction.start();
    coralTransaction.start();
    std::cout<<13<<std::endl;
    std::string tok2=metadata_svc.getToken("mytest2");
    std::cout<<14<<std::endl;
    coralTransaction.commit();
    //coralTransaction.commit();
    std::cout<<15<<std::endl;
    std::cout<<"got token2 "<<tok2<<std::endl;
    std::string newtok2="newtoken2";
    coralTransaction.start();
    //coralTransaction.start();
    std::cout<<"abp"<<std::endl;
    metadata_svc.replaceToken("mytest2",newtok2);
    coralTransaction.commit();
    //coralTransaction.commit();
    //coralTransaction.start();
    coralTransaction.start();
    std::string mytok2=metadata_svc.getToken("mytest2");
    std::cout<<"get back new tok2 "<<newtok2<<" "<<mytok2<<std::endl;
    std::cout<<"tag exists mytest2 "<<metadata_svc.hasTag("mytest2")<<std::endl;
    std::cout<<"tag exists crap "<<metadata_svc.hasTag("crap")<<std::endl;
    std::vector<std::string> alltags;
    metadata_svc.listAllTags(alltags);
    //coralTransaction.commit();
    coralTransaction.commit();
    std::copy (alltags.begin(),
	       alltags.end(),
	       std::ostream_iterator<std::string>(std::cout,"\n")
	       );
    conHandler.disconnectAll();
    delete session;
  }catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(std::exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Funny error"<<std::endl;
  }
}


