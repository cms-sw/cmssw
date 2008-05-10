/*
 *  See headers for a description
 *
 *  $Date: 2008/05/01 15:26:25 $
 *  $Revision: 1.2 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "RPCSourceHandler.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<iostream>
#include <fstream>
#include <time.h>

popcon::RpcData::RpcData(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","RpcData")), 
  host(pset.getUntrackedParameter<std::string>("host", "source db host")),
  user(pset.getUntrackedParameter<std::string>("user", "source username")),
  passw(pset.getUntrackedParameter<std::string>("passw", "source password")),
  Ohost(pset.getUntrackedParameter<std::string>("host", "dest db host")),
  Ouser(pset.getUntrackedParameter<std::string>("user", "dest username")),
  Opassw(pset.getUntrackedParameter<std::string>("passw", "dest password")),
  since(pset.getUntrackedParameter<int>("since", 1210341600)),
  logpath(pset.getUntrackedParameter<std::string>("logpath", "log file path")){
}

popcon::RpcData::~RpcData()
{
}

void popcon::RpcData::getNewObjects() {

  std::cout << "Reading log file: " << logpath << std::endl;
  std::string ltime;
  std::ifstream logfile;
  logfile.open(logpath.c_str());
  getline (logfile,ltime);
  std::cout << "last time data: " << ltime << std::endl;
 
  logfile.close();
  
  std::cout << "------- " << m_name 
	    << " - > getNewObjects\n" << "got offlineInfo for " 
	    << tagInfo().name << ", size " << tagInfo().size 
	    << ", last object valid since " 
	    << tagInfo().lastInterval.first << " till "
	    << tagInfo().lastInterval.second << " token "   
	    << tagInfo().lastPayloadToken << std::endl;
  
  coral::TimeStamp* mytime = new coral::TimeStamp;
  coral::TimeStamp now = mytime->now();

  time_t Ttll;
  Ttll = time (NULL);  
  tll = time (NULL);

  RPCFw caen ( host, user, passw ); // OMDS
  
  if (since > 0) {
     std::cout << std::endl << ">> User since time selection >> " << std::endl;
     snc = since;
  } else {
     std::cout <<  std::endl << ">> Since time from log file >> " << std::endl;
     snc = atoi(ltime.c_str());
  }

  struct tm * utctime;
  struct tm * loctime;
  time ( &Ttll );
  utctime = gmtime ( &Ttll );
  loctime = localtime ( &Ttll );
  printf ( ">> Current UTC Time Stamp is: %s", asctime (utctime) );
  printf ( ">> Local Time Stamp is: %s", asctime (loctime) );

  std::cout << ">> UTime = " << tll << "--> IOV :: since = " << snc << " :: till = " << tll << std::endl;

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

  int tll2 = tll + 1;
  int tll3 = tll + 2;

  m_to_transfer.push_back(std::make_pair((RPCdbData*)Idata,tll));
  m_to_transfer.push_back(std::make_pair((RPCdbData*)Vdata,tll2));
  m_to_transfer.push_back(std::make_pair((RPCdbData*)Sdata,tll3));

  std::cerr << "------- " << m_name << " - > 3 Objects writed." << std::endl;

  std::ofstream uplog;
  uplog.open(logpath.c_str(), std::ios::trunc);
  uplog << tll;
  uplog.close();

  std::cout << std::endl << ">> (Log file) " << logpath << " updated." << std::endl;
  
}



