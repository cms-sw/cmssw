/*
 *  See headers for a description
 *
 *  $Date: 2008/05/01 15:26:25 $
 *  $Revision: 1.1 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "RPCGasSH.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<iostream>
#include <fstream>
#include <time.h>

popcon::RpcGas::RpcGas(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","RpcGas")), 
  host(pset.getUntrackedParameter<std::string>("host", "source db host")),
  user(pset.getUntrackedParameter<std::string>("user", "source username")),
  passw(pset.getUntrackedParameter<std::string>("passw", "source password")),
  Ohost(pset.getUntrackedParameter<std::string>("host", "dest db host")),
  Ouser(pset.getUntrackedParameter<std::string>("user", "dest username")),
  Opassw(pset.getUntrackedParameter<std::string>("passw", "dest password")),
  since(pset.getUntrackedParameter<int>("since", 1210341600)),
  logpath(pset.getUntrackedParameter<std::string>("logpath", "log file path")){
}

popcon::RpcGas::~RpcGas()
{
}

void popcon::RpcGas::getNewObjects() {

  std::cout << "Reading log file: " << logpath << std::endl;
  std::string ltime;
  std::ifstream logfile;
  logfile.open(logpath.c_str());
  getline (logfile,ltime);
  std::cout << "last time data: " << ltime << std::endl;
 
  logfile.close();

  int max_since=0;
  max_since=(int)tagInfo().lastInterval.first;
  std::cout << "max_since : "  << max_since << std::endl;
    


  
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

  std::vector<RPCGas::GasItem> Gcheck;
  
  Gcheck = caen.createGAS(snc);
  
  // make an fill Gas object
  Gdata = new RPCGas();
  RPCGas::GasItem Ifill;
  std::vector<RPCGas::GasItem>::iterator Iit;
  for(Iit = Gcheck.begin(); Iit != Gcheck.end(); Iit++)
    {
      Ifill = *(Iit);
      Gdata->Gas_rpc.push_back(Ifill);
    }
  std::cout << "Incoming object size: " << Gcheck.size() << " >> Final object size: " << Gdata->Gas_rpc.size() << std::endl;


  m_to_transfer.push_back(std::make_pair((RPCGas*)Gdata,tll));

  std::cerr << "------- " << m_name << " - > 1 Objects writed." << std::endl;

  std::ofstream uplog;
  uplog.open(logpath.c_str(), std::ios::trunc);
  uplog << tll;
  uplog.close();

  std::cout << std::endl << ">> (Log file) " << logpath << " updated." << std::endl;
  
}



