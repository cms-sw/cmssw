/*
 *  See headers for a description
 *
 *  $Date: 2008/08/28 10:36:52 $
 *  $Revision: 1.5 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "CondTools/RPC/interface/RPCIDMapSH.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <sys/time.h>
#include "DataFormats/Provenance/interface/Timestamp.h"
#include<iostream>

popcon::RPCObPVSSmapData::RPCObPVSSmapData(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","RpcData")),
  host(pset.getUntrackedParameter<std::string>("host", "source db host")),
  user(pset.getUntrackedParameter<std::string>("user", "source username")),
  passw(pset.getUntrackedParameter<std::string>("passw", "source password")),
  m_since(pset.getUntrackedParameter<unsigned long long>("since",5)){
}

popcon::RPCObPVSSmapData::~RPCObPVSSmapData()
{
}

void popcon::RPCObPVSSmapData::getNewObjects() {

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
  ::timeval tv;
  gettimeofday(&tv,0);
  edm::Timestamp tstamp((unsigned long long)tv.tv_usec);
  std::cout << "Now ==> UNIX TIME = " << tstamp.value() << std::endl;
  utime = tstamp.value();
  //-----------------------------


  if (snc > 0) { niov = utime;} else { snc = m_since; niov = utime; }

  std::cout << "New IOV: since is = " << niov << std::endl;

  
  RPCFw caen ( host, user, passw );

  std::vector<RPCObPVSSmap::Item> IDMapcheck;


  IDMapcheck = caen.createIDMAP();


  IDMapdata = new RPCObPVSSmap();
  RPCObPVSSmap::Item Ifill;
  std::vector<RPCObPVSSmap::Item>::iterator Iit;
  for(Iit = IDMapcheck.begin(); Iit != IDMapcheck.end(); Iit++)
    {
      Ifill = *(Iit);
      IDMapdata->ObIDMap_rpc.push_back(Ifill);
    }
  std::cout << " >> Final object size: " << IDMapdata->ObIDMap_rpc.size() << std::endl;


  m_to_transfer.push_back(std::make_pair((RPCObPVSSmap*)IDMapdata,niov));
}
