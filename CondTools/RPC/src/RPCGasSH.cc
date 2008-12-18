/*
 *  See headers for a description
 *
 *  $Date: 2008/10/11 08:48:24 $
 *  $Revision: 1.6 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "CondTools/RPC/interface/RPCGasSH.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<iostream>

popcon::RpcObGasData::RpcObGasData(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","RpcData")),
  host(pset.getUntrackedParameter<std::string>("host", "source db host")),
  user(pset.getUntrackedParameter<std::string>("user", "source username")),
  passw(pset.getUntrackedParameter<std::string>("passw", "source password")),
  m_since(pset.getUntrackedParameter<unsigned long long>("since",5)){
}

popcon::RpcObGasData::~RpcObGasData()
{
}

void popcon::RpcObGasData::getNewObjects() {

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

  std::vector<RPCObGas::Item> Gascheck;


  Gascheck = caen.createGAS(snc);


  Gasdata = new RPCObGas();
  RPCObGas::Item Ifill;
  std::vector<RPCObGas::Item>::iterator Iit;
  for(Iit = Gascheck.begin(); Iit != Gascheck.end(); Iit++)
    {
      Ifill = *(Iit);
      Gasdata->ObGas_rpc.push_back(Ifill);
    }
  std::cout << " >> Final object size: " << Gasdata->ObGas_rpc.size() << std::endl;





/*
  Gasdata = new RPCObGas();
  RPCObGas::Item Ifill;

  for (int i = 0; i < 5; i++) {
  Ifill.dpid = niov*10+i;
  Ifill.flowin = niov*10+2*i;
  Ifill.flowout = niov*10+3*i;
  Ifill.day = niov*10+4*i;
  Ifill.time = niov*10+5*i;

  Gasdata->ObGas_rpc.push_back(Ifill);
  }
  std::cout << " >> Final object size: " << Gasdata->ObGas_rpc.size() << std::endl;
*/

  m_to_transfer.push_back(std::make_pair((RPCObGas*)Gasdata,niov));
}
