/*
 *  See headers for a description
 *
 *  $Date: 2008/07/17 16:33:22 $
 *  $Revision: 1.5 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "CondTools/RPC/interface/RPCSourceHandler.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<iostream>

popcon::RpcData::RpcData(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","RpcData")),
  host(pset.getUntrackedParameter<std::string>("host", "source db host")),
  user(pset.getUntrackedParameter<std::string>("user", "source username")),
  passw(pset.getUntrackedParameter<std::string>("passw", "source password")),
  Ohost(pset.getUntrackedParameter<std::string>("host", "dest db host")),
  Ouser(pset.getUntrackedParameter<std::string>("user", "dest username")),
  Opassw(pset.getUntrackedParameter<std::string>("passw", "dest password")),
  m_since(pset.getUntrackedParameter<unsigned long long>("since",5)){
}

popcon::RpcData::~RpcData()
{
}

void popcon::RpcData::getNewObjects() {

  std::cout << "------- " << m_name << " - > getNewObjects\n" 
	    << "got offlineInfo"<< tagInfo().name 
	    << ", size " << tagInfo().size << ", last object valid since " 
	    << tagInfo().lastInterval.first << " token "   
            << tagInfo().lastPayloadToken << std::endl;

  std::cout << " ------ last entry info regarding the payload (if existing): " 
	    << logDBEntry().usertext << "last record with the correct tag has been written in the db: "
	    << logDBEntry().destinationDB << std::endl; 
  
  snc = tagInfo().lastInterval.first;

  //--------------------------IOV
  std::string str;
  time_t t;
  t = time (NULL);
  std::stringstream ss;
  ss << t; ss >> str;
  std::cout << "Now ==> UNIX TIME = " << str << std::endl;
  utime = atoi (str.c_str());  
  //-----------------------------


  if (snc > 0) { niov = utime;} else { exit(1); snc = m_since; niov = utime; }

  std::cout << "New IOV: since is = " << niov << std::endl;

  
  RPCFw caen ( host, user, passw );

  std::vector<RPCdbData::Item> Icheck;
  std::vector<RPCdbData::Item> Vcheck;
  std::vector<RPCdbData::Item> Scheck;

  Icheck = caen.createIMON(snc);
  Vcheck = caen.createVMON(snc);
  Scheck = caen.createSTATUS(snc);
  
  Idata = new RPCdbData();
  RPCdbData::Item Ifill;
  std::vector<RPCdbData::Item>::iterator Iit;
  for(Iit = Icheck.begin(); Iit != Icheck.end(); Iit++)
    {
      Ifill = *(Iit);
      Idata->Imon_rpc.push_back(Ifill);
    }
  std::cout << " >> Final object size: " << Idata->Imon_rpc.size() << std::endl;


  Vdata = new RPCdbData();
  RPCdbData::Item Vfill;
  std::vector<RPCdbData::Item>::iterator Vit;
  for(Vit = Vcheck.begin(); Vit != Vcheck.end(); Vit++)
    {
      Vfill = *(Iit);
      Vdata->Vmon_rpc.push_back(Vfill);
    }
  std::cout << " >> Final object size: " << Vdata->Vmon_rpc.size() << std::endl;


  Sdata = new RPCdbData();
  RPCdbData::Item Sfill;
  std::vector<RPCdbData::Item>::iterator Sit;
  for(Sit = Scheck.begin(); Sit != Scheck.end(); Sit++)
    {
      Sfill = *(Iit);
      Sdata->Status_rpc.push_back(Sfill);
    }
  std::cout << " >> Final object size: " << Sdata->Status_rpc.size() << std::endl;


  m_to_transfer.push_back(std::make_pair((RPCdbData*)Idata,niov));
  m_to_transfer.push_back(std::make_pair((RPCdbData*)Vdata,niov+1));
  m_to_transfer.push_back(std::make_pair((RPCdbData*)Sdata,niov+2));

}
