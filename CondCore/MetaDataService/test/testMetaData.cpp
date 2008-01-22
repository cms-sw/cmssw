#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <iostream>
int main(){
  try{
    cond::DBSession* session=new cond::DBSession;
    session->open();
    cond::Connection myconnection("sqlite_file:meta.db",-1);
    myconnection.connect(session);
    cond::CoralTransaction& coralTransaction=myconnection.coralTransaction();
    cond::MetaData metadata_svc(coralTransaction);
    coralTransaction.start(false);
    std::string t1("token1");
    metadata_svc.addMapping("mytest1",t1);
    coralTransaction.commit();
    std::string t2("token2");
    coralTransaction.start(false);
    metadata_svc.addMapping("mytest2",t2);
    coralTransaction.commit();
    coralTransaction.start(true);
    std::string tok1=metadata_svc.getToken("mytest2");
    coralTransaction.commit();
    std::cout<<"got token1 "<<tok1<<std::endl;
    coralTransaction.start(true);
    std::string tok2=metadata_svc.getToken("mytest2");
    coralTransaction.commit();
    //coralTransaction.commit();
    std::cout<<"got token2 "<<tok2<<std::endl;
    std::string newtok2="newtoken2";
    coralTransaction.start(false);
    //coralTransaction.start();
    std::cout<<"abp"<<std::endl;
    metadata_svc.replaceToken("mytest2",newtok2);
    coralTransaction.commit();
    //coralTransaction.commit();
    //coralTransaction.start();
    coralTransaction.start(true);
    std::string mytok2=metadata_svc.getToken("mytest2");
    std::cout<<"get back new tok2 "<<newtok2<<" "<<mytok2<<std::endl;
    std::cout<<"tag exists mytest2 "<<metadata_svc.hasTag("mytest2")<<std::endl;
    std::cout<<"tag exists crap "<<metadata_svc.hasTag("crap")<<std::endl;
    std::vector<std::string> alltags;
    metadata_svc.listAllTags(alltags);
    coralTransaction.commit();
    myconnection.disconnect();
    std::copy (alltags.begin(),
	       alltags.end(),
	       std::ostream_iterator<std::string>(std::cout,"\n")
	       );
    
    delete session;
  }catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(std::exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Funny error"<<std::endl;
  }
}


