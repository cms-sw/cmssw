/*
 *  See headers for a description
 *
 *  $Date: 2008/02/15 12:15:03 $
 *  $Revision: 1.1 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "RPCSourceHandler.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<iostream>

popcon::RpcData::RpcData(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","RpcData")){
  host = pset.getUntrackedParameter<std::string>("host", "host");
  user = pset.getUntrackedParameter<std::string>("user", "user");
  passw = pset.getUntrackedParameter<std::string>("passw", "passw");
}

popcon::RpcData::~RpcData()
{
}

void popcon::RpcData::getNewObjects() {
   std::cerr << "------- " << m_name 
	     << " - > getNewObjects" << std::endl;

  //check whats already inside of database
  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;  

  coral::TimeStamp* time = new coral::TimeStamp;
  coral::TimeStamp now = time->now();
  int utime = TtoUT(now);

  snc = tagInfo().lastInterval.first;
  tll = utime;

  std::cout <<">> Time = "<<now.day()<<"/"<<now.month()<<"/"<<now.year()<<" "<<now.hour()<<":"<<now.minute()<<"."<<now.second()<< std::endl;

  std::cout << ">> UTime = " << utime << "--> IOV :: since = " << snc << " :: till = " << tll << std::endl;
      

  RPCFw caen ( host, user, passw ); // OMDS
  
  //  snc = 1163552461; // just for the first time
 
  std::vector<RPCdbData::Item> Icheck;
  std::vector<RPCdbData::Item> Vcheck;
  std::vector<RPCdbData::Item> Scheck;
  
  Icheck = caen.createIMON(snc);
  Vcheck = caen.createVMON(snc);
  Scheck = caen.createSTATUS(snc);
  
  // make an fill Imon object
  Idata = new RPCdbData();
  RPCdbData::Item Ifill;
  std::vector<RPCdbData::Item>::iterator Iit;
  for(Iit = Icheck.begin(); Iit != Icheck.end(); Iit++)
    {
      Ifill = *(Iit);
      Idata->Imon_rpc.push_back(Ifill);
    }
  std::cout << "Incoming object size: " << Icheck.size() << " >> Final object size: " << Idata->Imon_rpc.size() << std::endl;


  // make an fill Vmon object
  Vdata = new RPCdbData();
  RPCdbData::Item Vfill;
  std::vector<RPCdbData::Item>::iterator Vit;
  for(Vit = Vcheck.begin(); Vit != Vcheck.end(); Vit++)
  {
  Vfill = *(Vit);
  Vdata->Vmon_rpc.push_back(Vfill);
  }
  std::cout << "Incoming object size: " << Vcheck.size() << " >> Final object size: " << Vdata->Vmon_rpc.size() << std::endl;


  // make an fill Status object
  Sdata = new RPCdbData();
  RPCdbData::Item Sfill;
  std::vector<RPCdbData::Item>::iterator Sit;
  for(Sit = Scheck.begin(); Sit != Scheck.end(); Sit++)
  {
  Sfill = *(Sit);
  Sdata->Status_rpc.push_back(Sfill);
  }
  std::cout << "Incoming object size: " << Scheck.size() << " >> Final object size: " << Sdata->Status_rpc.size() << std::endl;


  m_to_transfer.push_back(std::make_pair((RPCdbData*)Idata,tll));
  m_to_transfer.push_back(std::make_pair((RPCdbData*)Vdata,snc));
  m_to_transfer.push_back(std::make_pair((RPCdbData*)Sdata,snc));

  std::cerr << "------- " << m_name << " - > getNewObjects" << std::endl;
  
}
