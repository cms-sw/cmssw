/*
 *  See headers for a description
 *
 *  $Date: 2008/12/30 10:09:38 $
 *  $Revision: 1.4 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "CondTools/RPC/interface/RPCFebmapSH.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<iostream>

popcon::RpcDataFebmap::RpcDataFebmap(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","RpcData")),
  host(pset.getUntrackedParameter<std::string>("host", "source db host")),
  user(pset.getUntrackedParameter<std::string>("user", "source username")),
  passw(pset.getUntrackedParameter<std::string>("passw", "source password")),
  m_since(pset.getUntrackedParameter<unsigned long long>("since",5)){
}

popcon::RpcDataFebmap::~RpcDataFebmap()
{
}

void popcon::RpcDataFebmap::getNewObjects() {

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

  std::vector<RPCObFebmap::Feb_Item> Febmapcheck;

  Febmapcheck = caen.createFEB(snc);

  Febdata = new RPCObFebmap();
  RPCObFebmap::Feb_Item Febfill;
  std::vector<RPCObFebmap::Feb_Item>::iterator Febit;
  for(Febit = Febmapcheck.begin(); Febit != Febmapcheck.end(); Febit++)
    {
      Febfill = *(Febit);
      Febdata->ObFebMap_rpc.push_back(Febfill);
    }
  std::cout << " >> Final object size: " << Febdata->ObFebMap_rpc.size() << std::endl;

  m_to_transfer.push_back(std::make_pair((RPCObFebmap*)Febdata,niov));

}

