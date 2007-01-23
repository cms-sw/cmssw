#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/Ref.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "testPayloadObj.h"
#include <iostream>
int main(){
  std::string sourceConnect("sqlite_file:source.db");
  std::string destConnect("sqlite_file:dest.db");
  std::string catalog("file:mycatalog.xml");
  try{
    cond::DBSession* session=new cond::DBSession(true);
    session->sessionConfiguration().setMessageLevel(cond::Error);
    session->open();
    cond::PoolStorageManager sourcedb(sourceConnect,catalog,session);
    sourcedb.connect();
    cond::IOVService iovmanager(sourcedb);
    cond::IOVEditor* editor=iovmanager.newIOVEditor();
    sourcedb.startTransaction(false);
    for(int i=0; i<5; ++i){
      std::cout<<"creating test payload obj"<<i<<std::endl;
      testPayloadObj* myobj=new testPayloadObj;
      for(int j=0; j<10; ++j){
	myobj->data.push_back(i+j);
      }
      cond::Ref<testPayloadObj> myobjRef(sourcedb,myobj);
      myobjRef.markWrite("testPayloadObjRcd");
      editor->insert(i+10, myobjRef.token());
    }
    std::string iovtoken=editor->token();
    std::cout<<"iov token "<<iovtoken<<std::endl;
    sourcedb.commit();
    
    cond::PoolStorageManager destdb(destConnect,catalog,session);
    destdb.connect();
    sourcedb.startTransaction(true);
    destdb.startTransaction(false);
    iovmanager.exportIOVWithPayload( destdb,
				     iovtoken,
				     "testPayloadObj" );
    sourcedb.commit();
    destdb.commit();
    sourcedb.disconnect();
    destdb.disconnect();
    session->close();
    delete editor;
    delete session;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
