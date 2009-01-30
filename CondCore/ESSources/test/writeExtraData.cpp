#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
//#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/Exception.h"
//#include "CondCore/DBCommon/interface/ConnectMode.h"
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
    cond::Time_t globalTill = cond::timeTypeSpecs[timetype].endValue;

    cond::DBSession* session=new cond::DBSession;
    session->configuration().setMessageLevel(cond::Error);
    cond::Connection myconnection("sqlite_file:extradata.db",-1);
    session->open();
    myconnection.connect(session);
    cond::PoolTransaction& pooldb=myconnection.poolTransaction();
    cond::IOVService iovmanager(pooldb);
    cond::IOVEditor* ioveditor=iovmanager.newIOVEditor();
    pooldb.start(false);
    ioveditor->create(timetype,globalTill);
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
      myref.markWrite("anotherPedestalsRcd");
      std::string payloadToken=myref.token();
      std::cout<<"payloadToken "<<payloadToken<<std::endl;
      ioveditor->append(cond::Time_t(2+2*i),payloadToken);
    }
    std::string mytoken=ioveditor->token();
    pooldb.commit();
    delete ioveditor;
 
    
    cond::CoralTransaction& coraldb=myconnection.coralTransaction();
    cond::MetaData metadata(coraldb);
    coraldb.start(false);
    metadata.addMapping("anothertag",mytoken,cond::runnumber);
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
