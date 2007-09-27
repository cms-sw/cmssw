#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondFormats/Calibration/interface/Pedestals.h"
int main(){
  try{
    cond::DBSession* session=new cond::DBSession;
    session->configuration().setMessageLevel(cond::Error);
    static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
    conHandler.registerConnection("mysqlite","sqlite_file:mytest.db",0);
    session->open();
    conHandler.connect(session);
    cond::Connection* myconnection=conHandler.getConnection("mysqlite");
    std::cout<<"myconnection "<<myconnection<<std::endl;
    cond::PoolTransaction& pooldb=myconnection->poolTransaction(false);
    cond::IOVService iovmanager(pooldb);
    cond::IOVEditor* ioveditor=iovmanager.newIOVEditor();
    pooldb.start();
    std::string mytestiovtoken;
    for(unsigned int i=0; i<3; ++i){ //inserting 3 payloads
      Pedestals* myped=new Pedestals;
      for(int ichannel=1; ichannel<=5; ++ichannel){
	Pedestals::Item item;
        item.m_mean=1.11*ichannel+i;
        item.m_variance=1.12*ichannel+i*2;
        myped->m_pedestals.push_back(item);
      }
      cond::TypedRef<Pedestals> myref(pooldb,myped);
      myref.markWrite("PedestalsRcd");
      std::string payloadToken=myref.token();
      ioveditor->insert(cond::Time_t(2+2*i),payloadToken);
    }
    //last one
    Pedestals* myped=new Pedestals;
    for(int ichannel=1; ichannel<=5; ++ichannel){
      Pedestals::Item item;
      item.m_mean=3.11*ichannel;
      item.m_variance=5.12*ichannel;
      myped->m_pedestals.push_back(item);
    }
    cond::TypedRef<Pedestals> myref(pooldb,myped);
    myref.markWrite("PedestalsRcd");
    std::string payloadToken=myref.token();
    ioveditor->insert(9999,payloadToken);
    mytestiovtoken=ioveditor->token();
    std::cout<<"mytest iov token "<<mytestiovtoken<<std::endl;
    pooldb.commit();
    delete ioveditor;
    
    std::string mypedestalsiovtoken;
    cond::IOVEditor* ioveditor2=iovmanager.newIOVEditor();
    pooldb.start();
    for(unsigned int i=0; i<2; ++i){ //inserting 3 payloads
      Pedestals* myped=new Pedestals;
      for(int ichannel=1; ichannel<=5; ++ichannel){
	Pedestals::Item item;
        item.m_mean=1.11*ichannel+i;
        item.m_variance=1.12*ichannel+i*2;
        myped->m_pedestals.push_back(item);
      }
      cond::TypedRef<Pedestals> myref(pooldb,myped);
      myref.markWrite("PedestalsRcd");
      std::string payloadToken=myref.token();
      ioveditor2->insert(cond::Time_t(5+2*i),payloadToken);
    }
    mypedestalsiovtoken=ioveditor2->token();
    std::cout<<"mytest iov token "<<mypedestalsiovtoken<<std::endl;
    pooldb.commit();
    delete ioveditor2;

    //
    ///I write different pedestals in another record
    //
    std::string anothermytestiovtoken;
    cond::IOVEditor* anotherioveditor=iovmanager.newIOVEditor();
    pooldb.start();
    for(unsigned int i=0; i<2; ++i){ //inserting 2 payloads to another Rcd
      Pedestals* myped=new Pedestals;
      for(int ichannel=1; ichannel<=3; ++ichannel){
	Pedestals::Item item;
        item.m_mean=1.11*ichannel+i;
        item.m_variance=1.12*ichannel+i*2;
        myped->m_pedestals.push_back(item);
      }
      cond::TypedRef<Pedestals> myref(pooldb,myped);
      myref.markWrite("anotherPedestalsRcd");
      std::string payloadToken=myref.token();
      anotherioveditor->insert(cond::Time_t(2+2*i),payloadToken);
    }
    anothermytestiovtoken=anotherioveditor->token();
    pooldb.commit();
    std::cout<<"anothermytest iov token "<<anothermytestiovtoken<<std::endl;
    delete anotherioveditor;
    
    cond::CoralTransaction& coraldb=myconnection->coralTransaction(false);
    cond::MetaData metadata(coraldb);
    coraldb.start();
    metadata.addMapping("mytest",mytestiovtoken);
    metadata.addMapping("mypedestals",mypedestalsiovtoken);
    metadata.addMapping("anothermytest",anothermytestiovtoken);
    coraldb.commit();
    delete session;
    session=0;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
