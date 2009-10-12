/*
 *  See headers for a description
 *
 *  $Date: 2008/12/12 20:20:41 $
 *  $Revision: 1.2 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "CondTools/RPC/interface/RPCVmonSH.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<iostream>

popcon::RpcDataV::RpcDataV(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","RpcData")),
  host(pset.getUntrackedParameter<std::string>("host", "source db host")),
  user(pset.getUntrackedParameter<std::string>("user", "source username")),
  passw(pset.getUntrackedParameter<std::string>("passw", "source password")),
  m_since(pset.getUntrackedParameter<unsigned long long>("since",5)){
}

popcon::RpcDataV::~RpcDataV()
{
}

void popcon::RpcDataV::getNewObjects() {

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

  std::vector<RPCObVmon::V_Item> Vcheck;

  Vcheck = caen.createVMON(snc);

  Vdata = new RPCObVmon();
  RPCObVmon::V_Item Vfill;
  std::vector<RPCObVmon::V_Item>::iterator Vit;
  for(Vit = Vcheck.begin(); Vit != Vcheck.end(); Vit++)
    {
      Vfill = *(Vit);
      Vdata->ObVmon_rpc.push_back(Vfill);
    }
  std::cout << " >> Final object size: " << Vdata->ObVmon_rpc.size() << std::endl;

  m_to_transfer.push_back(std::make_pair((RPCObVmon*)Vdata,niov));
}

