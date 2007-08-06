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
  std::string catalog("file:mycatalog.xml");
  try{
    cond::DBSession* session=new cond::DBSession;
    session->configuration().setMessageLevel(cond::Error);
    session->configuration().setAuthenticationMethod(cond::XML);
    static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
    conHandler.registerConnection("mysource","sqlite_file:source.db","file:mycatalog.xml",0);
    conHandler.registerConnection("mydest","sqlite_file:dest.db","file:mycatalog.xml",0);
    session->open();
    
    cond::PoolTransaction& sourcedb=conHandler.getConnection("mysource")->poolTransaction(false);
    cond::PoolTransaction& destdb=conHandler.getConnection("mydest")->poolTransaction(false);
    
    cond::IOVService iovmanager(sourcedb);
    cond::IOVEditor* editor=iovmanager.newIOVEditor();
    sourcedb.start();
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
    sourcedb.start();
    destdb.start();
    iovmanager.exportIOVWithPayload( destdb,
				     iovtoken,
				     "testPayloadObj" );
    sourcedb.commit();
    destdb.commit();
    delete editor;
    delete session;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
