/*
 *  See headers for a description
 *
 *  $Date: 2008/10/11 08:49:31 $
 *  $Revision: 1.1 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "CondTools/RPC/interface/RPCCondSH.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<iostream>

popcon::RpcData::RpcData(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","RpcData")),
  host(pset.getUntrackedParameter<std::string>("host", "source db host")),
  user(pset.getUntrackedParameter<std::string>("user", "source username")),
  passw(pset.getUntrackedParameter<std::string>("passw", "source password")),
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


  if (snc > 0) { niov = utime;} else { snc = m_since; niov = utime; }

  std::cout << "New IOV: since is = " << niov << std::endl;

  
  RPCFw caen ( host, user, passw );

  std::vector<RPCObCond::Item> Icheck;
  std::vector<RPCObCond::Item> Vcheck;
  std::vector<RPCObCond::Item> Scheck;
  std::vector<RPCObCond::Item> Tcheck;

  Icheck = caen.createIMON(snc);
  Vcheck = caen.createVMON(snc);
  Scheck = caen.createSTATUS(snc);
  Tcheck = caen.createT(snc);  

  Idata = new RPCObCond();
  RPCObCond::Item Ifill;
  std::vector<RPCObCond::Item>::iterator Iit;
  for(Iit = Icheck.begin(); Iit != Icheck.end(); Iit++)
    {
      Ifill = *(Iit);
      Idata->ObImon_rpc.push_back(Ifill);
    }
  std::cout << " >> Final object size: " << Idata->ObImon_rpc.size() << std::endl;


  Vdata = new RPCObCond();
  RPCObCond::Item Vfill;
  std::vector<RPCObCond::Item>::iterator Vit;
  for(Vit = Vcheck.begin(); Vit != Vcheck.end(); Vit++)
    {
      Vfill = *(Vit);
      Vdata->ObVmon_rpc.push_back(Vfill);
    }
  std::cout << " >> Final object size: " << Vdata->ObVmon_rpc.size() << std::endl;


  Sdata = new RPCObCond();
  RPCObCond::Item Sfill;
  std::vector<RPCObCond::Item>::iterator Sit;
  for(Sit = Scheck.begin(); Sit != Scheck.end(); Sit++)
    {
      Sfill = *(Sit);
      Sdata->ObStatus_rpc.push_back(Sfill);
    }
  std::cout << " >> Final object size: " << Sdata->ObStatus_rpc.size() << std::endl;


  Tdata = new RPCObCond();
  RPCObCond::Item Tfill;
  std::vector<RPCObCond::Item>::iterator Tit;
  for(Tit = Tcheck.begin(); Tit != Tcheck.end(); Tit++)
    {
      Tfill = *(Tit);
      Tdata->ObTemp_rpc.push_back(Tfill);
    }
  std::cout << " >> Final object size: " << Tdata->ObTemp_rpc.size() << std::endl;



  m_to_transfer.push_back(std::make_pair((RPCObCond*)Idata,niov));
  m_to_transfer.push_back(std::make_pair((RPCObCond*)Vdata,niov+1));
  m_to_transfer.push_back(std::make_pair((RPCObCond*)Sdata,niov+2));
  m_to_transfer.push_back(std::make_pair((RPCObCond*)Tdata,niov+3));
}

