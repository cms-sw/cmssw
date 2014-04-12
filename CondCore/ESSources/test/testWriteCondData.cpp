#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondFormats/Calibration/interface/Pedestals.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include <iostream>

int main(){
  try{
    // for runnumber
    cond::TimeType timetype = cond::runnumber;
    cond::Time_t globalTill = cond::timeTypeSpecs[timetype].endValue;
    edmplugin::PluginManager::Config config;
    edmplugin::PluginManager::configure(edmplugin::standard::config());

    cond::DbConnection connection;
    connection.configuration().setMessageLevel( coral::Error );
    connection.configuration().setAuthenticationPath(".");
    connection.configure();
    cond::DbSession session = connection.createSession();
    session.open( "sqlite_file:test.db" );

    cond::IOVEditor ioveditor( session );
    session.transaction().start(false);
    std::cout<<"globalTill value "<<globalTill<<std::endl;
    ioveditor.create(timetype,globalTill);
    for(unsigned int i=0; i<3; ++i){ //inserting 3 payloads
      boost::shared_ptr<Pedestals> myped( new Pedestals );
      for(int ichannel=1; ichannel<=5; ++ichannel){
        Pedestals::Item item;
        item.m_mean=1.11*ichannel+i;
        item.m_variance=1.12*ichannel+i*2;
        myped->m_pedestals.push_back(item);
      }
      std::string payloadToken = session.storeObject(myped.get(),"PedestalsRcd");
      ioveditor.append(cond::Time_t(2+2*i),payloadToken);
    }
    //last one
    boost::shared_ptr<Pedestals> myped( new Pedestals );
    for(int ichannel=1; ichannel<=5; ++ichannel){
      Pedestals::Item item;
      item.m_mean=3.11*ichannel;
      item.m_variance=5.12*ichannel;
      myped->m_pedestals.push_back(item);
    }
    std::string payloadToken = session.storeObject(myped.get(),"PedestalsRcd");
    ioveditor.append(9001,payloadToken);
    std::string iovtoken=ioveditor.token();
    std::cout<<"iov token "<<iovtoken<<std::endl;
    session.transaction().commit();
    //pooldb.disconnect();
    session.transaction().start(false);
    ioveditor.create(timetype, globalTill);
    boost::shared_ptr<Pedestals> p( new Pedestals );
    for(int ichannel=1; ichannel<=2; ++ichannel){
      Pedestals::Item item;
      item.m_mean=4.11*ichannel;
      item.m_variance=5.82*ichannel;
      p->m_pedestals.push_back(item);
    }
    std::string payloadToken2 = session.storeObject(p.get(),"PedestalsRcd");
    ioveditor.append(90001,payloadToken2);
    std::string pediovtoken=ioveditor.token();
    std::cout<<"iov token "<<pediovtoken<<std::endl;
    session.transaction().commit();
    //
    ///I write different pedestals in another record
    //
    cond::IOVEditor anotherioveditor( session );
    session.transaction().start(false);
    anotherioveditor.create(timetype,globalTill);
    for(unsigned int i=0; i<2; ++i){ //inserting 2 payloads to another Rcd
      boost::shared_ptr<Pedestals> myped( new Pedestals );
      for(int ichannel=1; ichannel<=3; ++ichannel){
        Pedestals::Item item;
        item.m_mean=1.11*ichannel+i;
        item.m_variance=1.12*ichannel+i*2;
        myped->m_pedestals.push_back(item);
      }
      std::string payloadToken = session.storeObject(myped.get(),"anotherPedestalsRcd");
      anotherioveditor.append(cond::Time_t(2+2*i),payloadToken);
    }
    std::string anotheriovtoken=anotherioveditor.token();
    std::cout<<"anotheriovtoken "<<anotheriovtoken<<std::endl;
    session.transaction().commit();
    
    cond::MetaData metadata(session);
    session.transaction().start(false);
    metadata.addMapping("mytest",iovtoken,cond::runnumber);
    metadata.addMapping("pedtag",pediovtoken,cond::runnumber);
    metadata.addMapping("anothermytest",anotheriovtoken,cond::runnumber);
    session.transaction().commit();
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
