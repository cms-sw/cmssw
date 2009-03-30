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
    cond::Connection myconnection("sqlite_file:test.db",0);  
    myconnection.connect(session);
    cond::PoolTransaction& pooldb=myconnection.poolTransaction();
    cond::IOVService iovmanager(pooldb);  
    cond::IOVEditor* editor=iovmanager.newIOVEditor();
    pooldb.start(false);
    unsigned int pos=0;
    editor->create(cond::timestamp,9);
    pos=editor->append(1,"pay01tok");
    std::cout<<"insertd 1 payload at position "<<pos<<std::endl;
    pos=editor->insert(20,"pay010tok");
    std::cout<<"inserted 20 payload at position "<<pos<<std::endl;
    pos=editor->insert(40, "pay021tok");
    std::cout<<"inserted 40 payload at position "<<pos<<std::endl;
    pos=editor->insert(60, "pay041tok");
    std::cout<<"inserted 60 payload at position "<<pos<<std::endl;
    pos=editor->append(120,"pay120tok");
    std::cout<<"inserted 120 payload at position "<<pos<<std::endl;
    pos=editor->append(140, "pay140tok");
    std::cout<<"inserted 140 payload at position "<<pos<<std::endl;
    pos=editor->append(160, "pay160tok");
    std::cout<<"inserted 160 payload at position "<<pos<<std::endl;
    pos=editor->append(170, "pay170tok");
    std::cout<<"inserted 170 payload at position "<<pos<<std::endl;
    try {
      pos=editor->insert(999999, "pay4tok");
      std::cout<<"shall not append payload at position "<<pos<<std::endl;
    }
    catch(const cond::Exception& er){
      std::cout<<"expected error "<<er.what()<<std::endl;
    }
    editor->updateClosure(300);
    pos=editor->insert(500, "pay301tok");
    std::cout<<"inserted 500 payload at position "<<pos<<std::endl;
    try {
      pos=editor->append(5, "pay5tok");
      std::cout<<"shall not append payload at position "<<pos<<std::endl;
    }
    catch(const cond::Exception& er){
      std::cout<<"expected error "<<er.what()<<std::endl;
    }
    try {
      pos=editor->insert(25, "pay5tok");
      std::cout<<"shall not insert payload at position "<<pos<<std::endl;
    }
    catch(const cond::Exception& er){
      std::cout<<"expected error "<<er.what()<<std::endl;
    }

    try {
      pos=editor->append(70, "pay5tok");
      std::cout<<"shall not apped payload at position "<<pos<<std::endl;
    }
    catch(const cond::Exception& er){
      std::cout<<"expected error "<<er.what()<<std::endl;
    }
    editor->updateClosure(400);

    
    // test freeInsert
    std::cout<<"\nfreeInsert "<<std::endl;
    pos=editor->freeInsert(5,"pay005tok");
    std::cout<<"inserted 5 payload at position "<<pos<<std::endl;
    pos=editor->freeInsert(12,"pay012tok");
    std::cout<<"inserted 12 payload at position "<<pos<<std::endl;
    pos=editor->freeInsert(50,"pay050tok");
    std::cout<<"inserted 50 payload at position "<<pos<<std::endl;
    pos=editor->freeInsert(119,"pay119tok");
    std::cout<<"inserted 119 payload at position "<<pos<<std::endl;
    pos=editor->freeInsert(141,"pay141tok");
    std::cout<<"inserted 141 payload at position "<<pos<<std::endl;
    pos=editor->freeInsert(142,"pay142tok");
    std::cout<<"inserted 142 payload at position "<<pos<<std::endl;
    pos=editor->freeInsert(399,"pay399tok");
    std::cout<<"inserted 399 payload at position "<<pos<<std::endl;
    pos=editor->freeInsert(521,"pay521tok");
    std::cout<<"inserted 521 payload at position "<<pos<<std::endl;
    try {
      pos=editor->freeInsert(5, "payNOtok");
      std::cout<<"shall not insert 5 payload at position "<<pos<<std::endl;
    }
    catch(const cond::Exception& er){
      std::cout<<"expected error "<<er.what()<<std::endl;
    }
    try {
      pos=editor->freeInsert(10, "payNOtok");
      std::cout<<"shall not insert 10 payload at position "<<pos<<std::endl;
    }
    catch(const cond::Exception& er){
      std::cout<<"expected error "<<er.what()<<std::endl;
    }
    try {
      pos=editor->freeInsert(21, "payNOtok");
      std::cout<<"shall not insert 21 payload at position "<<pos<<std::endl;
    }
    catch(const cond::Exception& er){
      std::cout<<"expected error "<<er.what()<<std::endl;
    }
    try {
      pos=editor->freeInsert(120, "payNOtok");
      std::cout<<"shall not insert 121 payload at position "<<pos<<std::endl;
    }
    catch(const cond::Exception& er){
      std::cout<<"expected error "<<er.what()<<std::endl;
    }
    try {
      pos=editor->freeInsert(50, "payNOtok");
      std::cout<<"shall not inser 50 payload at position "<<pos<<std::endl;
    }
    catch(const cond::Exception& er){
      std::cout<<"expected error "<<er.what()<<std::endl;
    }
 

    editor->updateClosure(900);
    

    std::cout<<"delete entry "<<std::endl;

    /*
    try {
      pos=editor->deleteEntry(5);
      std::cout<<"shall not delete payload at position "<<pos<<std::endl;
    }
    catch(const cond::Exception& er){
      std::cout<<"expected error "<<er.what()<<std::endl;
    }

    try {
      pos=editor->deleteEntry(5000);
      std::cout<<"shall not delete payload at position "<<pos<<std::endl;
    }
    catch(const cond::Exception& er){
      std::cout<<"expected error "<<er.what()<<std::endl;
    }
    

    pos=editor->append(593, "pay593tok");
    std::cout<<"inserted 193 payload at position "<<pos<<std::endl;
 

    pos=editor->deleteEntry(593);
    std::cout<<"deleted entry 593 payload at position "<<pos<<std::endl;

    pos=editor->deleteEntry(160);
    std::cout<<"deleted entry 160 payload at position "<<pos<<std::endl;
    pos=editor->deleteEntry(11);
    std::cout<<"deleted entry 11 payload at position "<<pos<<std::endl;
    pos=editor->deleteEntry(45);
    std::cout<<"deleted entry 45 payload at position "<<pos<<std::endl;
    pos=editor->deleteEntry(141);
    std::cout<<"deleted entry 141 payload at position "<<pos<<std::endl;
    */


    std::string token=editor->token();
    std::cout<<"iov token "<<token<<std::endl;
    pooldb.commit();
    delete editor;

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
    delete session;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
