#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"

#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVProxy.h"


#include <iostream>
int main(){
  edmplugin::PluginManager::Config config;
  try{
    edmplugin::PluginManager::configure(edmplugin::standard::config());
    cond::DbConnection connection;
    connection.configuration().setPoolAutomaticCleanUp( false );
    connection.configure();
    cond::DbSession pooldb = connection.createSession();
    pooldb.open("sqlite_file:test.db");
    std::string token;
    {
      cond::IOVEditor editor(pooldb);
      cond::DbScopedTransaction transaction(pooldb);
      transaction.start(false);
      unsigned int pos=0;
      editor.createIOVContainerIfNecessary();
      editor.create(cond::timestamp,9);
      pos=editor.append(1,"pay01tok");
      std::cout<<"insertd 1 payload at position "<<pos<<std::endl;
      pos=editor.insert(20,"pay010tok");
      std::cout<<"inserted 20 payload at position "<<pos<<std::endl;
      pos=editor.insert(40, "pay021tok");
      std::cout<<"inserted 40 payload at position "<<pos<<std::endl;
      pos=editor.insert(60, "pay041tok");
      std::cout<<"inserted 60 payload at position "<<pos<<std::endl;
      pos=editor.append(120,"pay120tok");
      std::cout<<"inserted 120 payload at position "<<pos<<std::endl;
      pos=editor.append(140, "pay140tok");
      std::cout<<"inserted 140 payload at position "<<pos<<std::endl;
      pos=editor.append(160, "pay160tok");
      std::cout<<"inserted 160 payload at position "<<pos<<std::endl;
      pos=editor.append(170, "pay170tok");
      std::cout<<"inserted 170 payload at position "<<pos<<std::endl;
      try {
	pos=editor.insert(999999, "pay4tok");
	std::cout<<"shall not append payload at position "<<pos<<std::endl;
      }
      catch(const cond::Exception& er){
	std::cout<<"expected error "<<er.what()<<std::endl;
      }
      editor.updateClosure(300);
      pos=editor.insert(500, "pay301tok");
      std::cout<<"inserted 500 payload at position "<<pos<<std::endl;
      try {
	pos=editor.append(5, "pay5tok");
	std::cout<<"shall not append payload at position "<<pos<<std::endl;
      }
      catch(const cond::Exception& er){
	std::cout<<"expected error "<<er.what()<<std::endl;
      }
      try {
	pos=editor.insert(25, "pay5tok");
	std::cout<<"shall not insert payload at position "<<pos<<std::endl;
      }
      catch(const cond::Exception& er){
	std::cout<<"expected error "<<er.what()<<std::endl;
      }
      
      try {
	pos=editor.append(70, "pay5tok");
	std::cout<<"shall not apped payload at position "<<pos<<std::endl;
      }
      catch(const cond::Exception& er){
	std::cout<<"expected error "<<er.what()<<std::endl;
      }
      editor.updateClosure(400);
      
      
      // test freeInsert
      std::cout<<"\nfreeInsert "<<std::endl;
      pos=editor.freeInsert(5,"pay005tok");
      std::cout<<"inserted 5 payload at position "<<pos<<std::endl;
      pos=editor.freeInsert(12,"pay012tok");
      std::cout<<"inserted 12 payload at position "<<pos<<std::endl;
      pos=editor.freeInsert(50,"pay050tok");
      std::cout<<"inserted 50 payload at position "<<pos<<std::endl;
      pos=editor.freeInsert(51,"pay051tok");
      std::cout<<"inserted 51 payload at position "<<pos<<std::endl;
      pos=editor.freeInsert(52,"pay052tok");
      std::cout<<"inserted 52 payload at position "<<pos<<std::endl;
      pos=editor.freeInsert(119,"pay119tok");
      std::cout<<"inserted 119 payload at position "<<pos<<std::endl;
      pos=editor.freeInsert(141,"pay141tok");
      std::cout<<"inserted 141 payload at position "<<pos<<std::endl;
      pos=editor.freeInsert(142,"pay142tok");
      std::cout<<"inserted 142 payload at position "<<pos<<std::endl;
      pos=editor.freeInsert(399,"pay399tok");
      std::cout<<"inserted 399 payload at position "<<pos<<std::endl;
      pos=editor.freeInsert(521,"pay521tok");
      std::cout<<"inserted 521 payload at position "<<pos<<std::endl;
      try {
	pos=editor.freeInsert(5, "payNOtok");
	std::cout<<"shall not insert 5 payload at position "<<pos<<std::endl;
      }
      catch(const cond::Exception& er){
	std::cout<<"expected error "<<er.what()<<std::endl;
      }
      try {
	pos=editor.freeInsert(10, "payNOtok");
	std::cout<<"shall not insert 10 payload at position "<<pos<<std::endl;
      }
      catch(const cond::Exception& er){
	std::cout<<"expected error "<<er.what()<<std::endl;
      }
      try {
	pos=editor.freeInsert(21, "payNOtok");
	std::cout<<"shall not insert 21 payload at position "<<pos<<std::endl;
      }
      catch(const cond::Exception& er){
	std::cout<<"expected error "<<er.what()<<std::endl;
      }
      try {
	pos=editor.freeInsert(120, "payNOtok");
	std::cout<<"shall not insert 121 payload at position "<<pos<<std::endl;
      }
      catch(const cond::Exception& er){
	std::cout<<"expected error "<<er.what()<<std::endl;
      }
      try {
	pos=editor.freeInsert(50, "payNOtok");
	std::cout<<"shall not inser 50 payload at position "<<pos<<std::endl;
      }
      catch(const cond::Exception& er){
	std::cout<<"expected error "<<er.what()<<std::endl;
      }
      
      
      
      std::cout<<"delete entry "<<std::endl;
      
      
      pos=editor.append(20100, "payNOtok");
      std::cout<<"inserted 20100 payload at position "<<pos<<std::endl;
      pos=editor.append(20123, "payNOtok");
      std::cout<<"inserted 20123 payload at position "<<pos<<std::endl;
      
      // does not work....
      //  pos=editor.truncate();
      //std::cout<<"truncate. new last position "<<pos<<std::endl;
      
      
      token=editor.proxy().token();
      std::cout<<"iov token "<<token<<std::endl;
      transaction.commit();
    }
    
    {
      cond::IOVEditor editor(pooldb, token);
      cond::DbScopedTransaction transaction(pooldb);
      transaction.start(false);
      unsigned int pos=0;  
      pos=editor.truncate();
      std::cout<<"truncate. new last position "<<pos<<std::endl;
      pos=editor.truncate();
      std::cout<<"truncate. new last position "<<pos<<std::endl;
      
      editor.updateClosure(900);
      transaction.commit();
    }
    
    {
      cond::IOVEditor editor(pooldb,token);
      cond::DbScopedTransaction transaction(pooldb);
      transaction.start(false);
      unsigned int pos=0;
      pos=editor.append(1345, "pay1345tok");
      std::cout<<"inserted 1345 payload at position "<<pos<<std::endl;
      editor.updateClosure(1345);
      pos=editor.append(1346, "pay1346tok");
      std::cout<<"inserted 1346 payload at position "<<pos<<std::endl;
      editor.updateClosure(1346);
      transaction.commit();
    }
    

    cond::IOVProxy iov( pooldb, token );
    for ( cond::IOVProxy::const_iterator it =iov.begin(); it!=iov.end(); ++it) {
      std::cout<<"payloadToken "<<it->token();
      std::cout<<", since "<<it->since();
      std::cout<<", till "<<it->till()<<std::endl;
    }

  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
