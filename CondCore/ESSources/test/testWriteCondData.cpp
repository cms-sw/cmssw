#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondFormats/Calibration/interface/Pedestals.h"
int main(){
  try{
    // for runnumber
    cond::TimeType timetype = cond::runnumber;
    cond::Time_t globalSince = cond::timeTypeSpecs[timetype].beginValue;

    cond::DBSession* session=new cond::DBSession;
    session->configuration().setMessageLevel(cond::Error);
    //cond::Connection myconnection("oracle://cms_orcoff_prep/CMS_COND_PRESH",0);
    cond::Connection myconnection("sqlite_file:test.db",0);
    session->open();
    myconnection.connect(session);
    cond::PoolTransaction& pooldb=myconnection.poolTransaction();
    cond::IOVService iovmanager(pooldb);
    cond::IOVEditor* ioveditor=iovmanager.newIOVEditor();
    pooldb.start(false);
    std::cout<<"globalsince value "<<globalSince<<std::endl;
    ioveditor->create(globalSince,timetype);
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
    std::string iovtoken=ioveditor->token();
    std::cout<<"iov token "<<iovtoken<<std::endl;
    pooldb.commit();
    //pooldb.disconnect();
    //delete ioveditor;
    pooldb.start(false);
    ioveditor=iovmanager.newIOVEditor();
    ioveditor->create(globalSince,timetype);
    Pedestals* p=new Pedestals;
    for(int ichannel=1; ichannel<=2; ++ichannel){
      Pedestals::Item item;
      item.m_mean=4.11*ichannel;
      item.m_variance=5.82*ichannel;
      p->m_pedestals.push_back(item);
    }
    cond::TypedRef<Pedestals> m(pooldb,p);
    m.markWrite("PedestalsRcd");
    std::string payloadToken2=m.token();
    ioveditor->insert(9999,payloadToken2);
    std::string pediovtoken=ioveditor->token();
    std::cout<<"iov token "<<pediovtoken<<std::endl;
    pooldb.commit();
    delete ioveditor;
    //
    ///I write different pedestals in another record
    //
    cond::IOVEditor* anotherioveditor=iovmanager.newIOVEditor();
    pooldb.start(false);
    anotherioveditor->create(globalSince,timetype);
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
    std::string anotheriovtoken=anotherioveditor->token();
    std::cout<<"anotheriovtoken "<<anotheriovtoken<<std::endl;
    pooldb.commit();
    delete anotherioveditor;
    
    cond::CoralTransaction& coraldb=myconnection.coralTransaction();
    cond::MetaData metadata(coraldb);
    coraldb.start(false);
    metadata.addMapping("mytest",iovtoken,cond::runnumber);
    metadata.addMapping("pedtag",pediovtoken,cond::runnumber);
    metadata.addMapping("anothermytest",anotheriovtoken,cond::runnumber);
    coraldb.commit();
    myconnection.disconnect();
    delete session;
    session=0;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
