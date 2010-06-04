#include "CondCore/DBCommon/interface/DBWriter.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/ServiceLoader.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "CondCore/IOVService/interface/IOV.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include <string>
#include <iostream>
int main(){
  std::string db ("sqlite_file:test.db");
  cond::ServiceLoader* loader=new cond::ServiceLoader;
  //loader->loadMessageService(cond::Info);
  loader->loadMessageService(cond::Error);
  cond::DBSession* session=new cond::DBSession(db);
  // access metadata first
  std::cout << "Quering (empty yet) metadata" << std::endl;
  try {
    cond::MetaData metadata_svc(db, *loader);
    metadata_svc.connect();
    metadata_svc.addMapping ("dummy", "empty_token");
    std::string token = metadata_svc.getToken("mytest1");
    metadata_svc.disconnect();
  }catch(std::exception& er){
    std::cout<<"1"<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"1"<<"Funny error"<<std::endl;
  }
  std::cout << "Writing objects" << std::endl;

  try{
    session->connect(cond::ReadWriteCreate);
    cond::DBWriter pwriter(*session, "HcalPedestals");
    cond::DBWriter iovwriter(*session, "IOV");
    session->startUpdateTransaction();
    ///init iov sequence
    cond::IOV* initiov=new cond::IOV;
    std::vector<std::string> pedtoks;
    for(int i=0; i<4;++i){
      HcalPedestals* ped=new HcalPedestals;
      for(int ichannel=1; ichannel<=5; ++ichannel){
	HcalDetId id (HcalBarrel, ichannel, 1, 1); // HB eta=ichannel, phi=1
	ped->addValue (id, 5, 5, 5, 5);
      }
      ped->sort ();
      std::string pedtok=pwriter.markWrite<HcalPedestals>(ped);
      pedtoks.push_back(pedtok);
      if(i<2){
	initiov->iov.insert(std::make_pair(1,pedtok));
      }else{
	initiov->iov.insert(std::make_pair(2,pedtok));
      }
    }
    iovwriter.markWrite<cond::IOV>(initiov);
    session->commit();
    //create a new iov sequence
    session->startUpdateTransaction();
    cond::IOV* bestiov=new cond::IOV;
    int counter=0;
    for(std::vector<std::string>::iterator it=pedtoks.begin(); 
	it!=pedtoks.end(); ++it){
      ++counter;
      if(counter<3){
	bestiov->iov.insert(std::make_pair(1,*it));
      }else{
	bestiov->iov.insert(std::make_pair(2,*it));
      }
    }
    std::string bestiovtok=iovwriter.markWrite<cond::IOV>(bestiov);
    session->commit();
    session->disconnect();

    // put metadata
    cond::MetaData metadata_svc(db, *loader);
    metadata_svc.connect();
    metadata_svc.addMapping("mytest1",bestiovtok);
    metadata_svc.disconnect();
  }catch(cond::Exception& er){
    std::cout<<"2"<<er.what()<<std::endl;
  }catch(std::exception& er){
    std::cout<<"2"<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"2"<<"Funny error"<<std::endl;
  }
  delete session;
  delete loader;
}

