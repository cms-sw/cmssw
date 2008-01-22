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
    editor->insert(20,"pay1tok");
    editor->insert(40, "pay2tok");
    editor->insert(60, "pay3tok");
    editor->insert(999999, "pay4tok");
    editor->append(70, "pay5tok");
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
