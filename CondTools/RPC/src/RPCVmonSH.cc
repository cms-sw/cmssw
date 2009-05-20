/*
 *  See headers for a description
 *
 *  $Date: 2008/12/30 10:09:38 $
 *  $Revision: 1.4 $
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
  m_since(pset.getUntrackedParameter<unsigned long long>("since",5)),
  m_till(pset.getUntrackedParameter<unsigned long long>("till",5)){

}

popcon::RpcDataV::~RpcDataV()
{
}

void popcon::RpcDataV::getNewObjects() {

  std::cout << "------- " << m_name << " - > getNewObjects\n" 
	    << "got offlineInfo "<< tagInfo().name 
	    << ", size " << tagInfo().size << ", last object valid since " 
	    << tagInfo().lastInterval.first << " token "   
            << tagInfo().lastPayloadToken << std::endl;

  std::cout << " ------ last entry info regarding the payload (if existing): " 
	    << logDBEntry().usertext << "last record with the correct tag has been written in the db: "
	    << logDBEntry().destinationDB << std::endl; 
  
  snc = tagInfo().lastInterval.first;

   std::cout << std::endl << "=============================================" << std::endl;
   std::cout << std::endl << "===================  VMON  ==================" << std::endl;
   std::cout << std::endl << "=============================================" << std::endl << std::endl;

   if (snc == 0) snc = m_since; 
   if (m_till > snc) std::cout << ">> Range mode [" << snc << ", " << m_till << "]" << std::endl;
      else std::cout << ">> Append mode [" << m_since << ", now]" << std::endl;

   std::cout << std::endl << "=============================================" << std::endl << std::endl;
   

  
  RPCFw caen ( host, user, passw );

  std::vector<RPCObVmon::V_Item> Vcheck;
  Vcheck = caen.createVMON(snc, m_till);

  Vdata = new RPCObVmon();
  RPCObVmon::V_Item Vfill;
  std::vector<RPCObVmon::V_Item>::iterator Vit;
  for(Vit = Vcheck.begin(); Vit != Vcheck.end(); Vit++)
    {
      Vfill = *(Vit);
      Vdata->ObVmon_rpc.push_back(Vfill);
    }
  std::cout << " >> Final object size: " << Vdata->ObVmon_rpc.size() << std::endl;

   if (Vdata->ObVmon_rpc.size() > 0) {
     niov = caen.N_IOV;
     std::cout << "===> New IOV: since is = " << niov << std::endl;
   } else {
     niov = snc + 1;
     std::cout << "NO DATA TO BE STORED" << std::endl;
   }
  m_to_transfer.push_back(std::make_pair((RPCObVmon*)Vdata,niov));
}

