#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "testPayloadObj.h"
#include <iostream>
int main(){
  std::string sourceConnect("sqlite_file:source.db");
  std::string destConnect("sqlite_file:dest.db");
  try{
    cond::DBSession* session=new cond::DBSession;
    session->configuration().setMessageLevel(cond::Error);
    session->configuration().setAuthenticationMethod(cond::XML);
    static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
    conHandler.registerConnection("mysource","sqlite_file:source.db",0);
    conHandler.registerConnection("mydest","sqlite_file:dest.db",0);
    session->open();
    conHandler.connect(session);
    cond::PoolTransaction& sourcedb=conHandler.getConnection("mysource")->poolTransaction();
    cond::PoolTransaction& destdb=conHandler.getConnection("mydest")->poolTransaction();
    
    cond::IOVService iovmanager(sourcedb);
    cond::IOVEditor* editor=iovmanager.newIOVEditor();
    sourcedb.start(false);
    editor->create(1,cond::timestamp);
    for(int i=0; i<5; ++i){
      std::cout<<"creating test payload obj"<<i<<std::endl;
      testPayloadObj* myobj=new testPayloadObj;
      for(int j=0; j<10; ++j){
	myobj->data.push_back(i+j);
      }
      cond::TypedRef<testPayloadObj> myobjRef(sourcedb,myobj);
      myobjRef.markWrite("testPayloadObjRcd");
      editor->insert(i+10, myobjRef.token());
    }
    std::string iovtoken=editor->token();
    std::cout<<"iov token "<<iovtoken<<std::endl;
    sourcedb.commit();
    std::cout<<"source db created "<<std::endl;
    sourcedb.start(true);
    std::cout<<"source db started "<<std::endl;
    destdb.start(false);
    std::cout<<"dest db started "<<std::endl;
    iovmanager.exportIOVWithPayload( destdb,
				     iovtoken);
    destdb.commit();
    std::cout<<"destdb committed"<<std::endl;
    sourcedb.commit();
    std::cout<<"source db committed"<<std::endl;
    delete editor;
    std::cout<<"editor deleted"<<std::endl;
    delete session;
    std::cout<<"session deleted"<<std::endl;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
