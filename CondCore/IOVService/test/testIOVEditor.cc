#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include <iostream>
int main(){
  try{
    cond::DBSession* session=new cond::DBSession;
    session->open();
    cond::Connection myconnection("sqlite_file:test.db",0);  
    myconnection.connect(session);
    cond::PoolTransaction& pooldb=myconnection.poolTransaction();
    cond::IOVService iovmanager(pooldb);  
    cond::IOVEditor* editor=iovmanager.newIOVEditor();
    pooldb.start(false);
    unsigned int pos=0;
    pos=editor->insert(20,"pay1tok");
    std::cout<<"inserted 20 payload at position "<<pos<<std::endl;
    pos=editor->insert(40, "pay2tok");
    std::cout<<"inserted 40 payload at position "<<pos<<std::endl;
    pos=editor->insert(60, "pay3tok");
    std::cout<<"inserted 60 payload at position "<<pos<<std::endl;
    pos=editor->insert(999999, "pay4tok");
    std::cout<<"inserted 999999 payload at position "<<pos<<std::endl;
    pos=editor->append(70, "pay5tok");
    std::cout<<"appened payload at position "<<pos<<std::endl;
    editor->updateClosure(999997);
    std::string token=editor->token();
    std::cout<<"iov token "<<token<<std::endl;
    //cond::IOVEditor* bomber=iovmanager.newIOVEditor(token);
    //bomber->deleteEntries();
    pooldb.commit();
    myconnection.disconnect();
    delete editor;
    delete session;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
