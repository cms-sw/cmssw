#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVIterator.h"

#include <iostream>
int main(){
  try{
    cond::DBSession* session=new cond::DBSession;
    session->open();
    cond::Connection myconnection("sqlite_file:testIOVReplaceInterval.db",0);  
    myconnection.connect(session);
    cond::PoolTransaction& pooldb=myconnection.poolTransaction();
    cond::IOVService iovmanager(pooldb);  
    cond::IOVEditor* editor=iovmanager.newIOVEditor();
    pooldb.start(false);
    unsigned int pos=0;
    editor->create(10,cond::timestamp);
    pos=editor->insert(10,"pay010tok");
    std::cout<<"inserted payload 10 at position "<<pos<<std::endl;
    pos=editor->replaceInterval(11,11,"pay011tok");
    std::cout<<"inserted payload 11 at position "<<pos<<std::endl;
    pos=editor->replaceInterval(12,12,"pay012tok");
    std::cout<<"inserted payload 12 at position "<<pos<<std::endl;
    pos=editor->replaceInterval(15,60,"pay015-60tok");
    std::cout<<"inserted payload 15 at position "<<pos<<std::endl;
    pos=editor->replaceInterval(30,35,"pay030-35tok");
    std::cout<<"inserted payload 30 at position "<<pos<<std::endl;
    pos=editor->replaceInterval(32,32,"pay032tok");
    std::cout<<"inserted payload 32 at position "<<pos<<std::endl;
    pos=editor->replaceInterval(90,90,"pay090tok");
    std::cout<<"inserted payload 90 at position "<<pos<<std::endl;
    pos=editor->replaceInterval(91,91,"pay091tok");
    std::cout<<"inserted payload 91 at position "<<pos<<std::endl;
    pos=editor->replaceInterval(92,100,"pay092-100tok");
    std::cout<<"inserted payload 92 at position "<<pos<<std::endl;

    {    
      std::string token=editor->token();
      cond::IOVIterator* it=iovmanager.newIOVIterator(token);
      std::cout<<"forward iterator "<<std::endl;
      pooldb.start(true);
      while( it->next() ){
	std::cout<<"payloadToken "<<it->payloadToken();
	std::cout<<", since "<<it->validity().first;
	std::cout<<", till "<<it->validity().second<<std::endl;
      }
      delete it;
    }

    pos=editor->replaceInterval(10,10,"pay010newtok");
    std::cout<<"inserted new payload 10 at position "<<pos<<std::endl;
    pos=editor->replaceInterval(12,12,"pay012newtok");
    std::cout<<"inserted new payload 12 at position "<<pos<<std::endl;
    pos=editor->replaceInterval(10,10,"pay010newtok");
    std::cout<<"inserted new payload 10 at position "<<pos<<std::endl;
    pos=editor->replaceInterval(15,15,"pay015newtok");
    std::cout<<"inserted new payload 15 at position "<<pos<<std::endl;
    pos=editor->replaceInterval(60,60,"pay060newtok");
    std::cout<<"inserted new payload 60 at position "<<pos<<std::endl;

    pos=editor->replaceInterval(85,95,"pay085-95tok");
    std::cout<<"inserted payload 85 at position "<<pos<<std::endl;
    pos=editor->replaceInterval(5,5,"pay005newtok");
    std::cout<<"inserted new payload 5 at position "<<pos<<std::endl;


    std::string token=editor->token();
    std::cout<<"iov token "<<token<<std::endl;
    cond::IOVIterator* it=iovmanager.newIOVIterator(token);
    std::cout<<"forward iterator "<<std::endl;
    pooldb.start(true);
    while( it->next() ){
      std::cout<<"payloadToken "<<it->payloadToken();
      std::cout<<", since "<<it->validity().first;
      std::cout<<", till "<<it->validity().second<<std::endl;
    }
    delete it;
    
    //cond::IOVEditor* bomber=iovmanager.newIOVEditor(token);
    //bomber->deleteEntries();
    pooldb.commit();
    myconnection.disconnect();
    delete editor;
    delete session;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
    return -1;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
    return -1;
  }
  return 0;
}
