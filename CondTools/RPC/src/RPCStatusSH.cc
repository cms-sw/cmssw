/*
 *  See headers for a description
 *
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "CondTools/RPC/interface/RPCStatusSH.h"
#include<sstream>
#include<iostream>

popcon::RpcDataS::RpcDataS(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","RpcData")),
  host(pset.getUntrackedParameter<std::string>("host", "source db host")),
  user(pset.getUntrackedParameter<std::string>("user", "source username")),
  passw(pset.getUntrackedParameter<std::string>("passw", "source password")),
  m_first(pset.getUntrackedParameter<bool>("first",false)),
  m_since(pset.getUntrackedParameter<unsigned long long>("since",1)),
  m_range(pset.getUntrackedParameter<unsigned long long>("range",72000)){
}

popcon::RpcDataS::~RpcDataS()
{
}

void popcon::RpcDataS::getNewObjects() {
  
  std::cout << "------- " << m_name << " - > getNewObjects\n" 
	    << "got offlineInfo "<< tagInfo().name 
	    << ", size " << tagInfo().size << ", last object valid since " 
	    << tagInfo().lastInterval.first << " token "   
            << tagInfo().lastPayloadToken << std::endl;
  
  std::cout << " ------ last entry info regarding the payload (if existing): " 
	    << logDBEntry().usertext << "last record with the correct tag has been written in the db: "
	    << logDBEntry().destinationDB << std::endl; 
  
  // Get from the logDBEntry the till unix time query and the number of values stored
  std::stringstream is;
  std::string a = logDBEntry().usertext;
  unsigned int l = a.find('>');
  unsigned int preTill;
  
  if (l < a.size()){
    is <<a.substr(l+3,a.npos);
    std::string b1,b2; 
    unsigned int nVals;
    is >>b1>>nVals>>b2>>preTill;
    std::cout <<" Unix Time of the Prev Till "<<preTill<<std::endl;
  }else{
    std::cout <<" No infos from usertext in logDB"<<std::endl;
  }

  if (!m_first) {
    m_since = preTill;
  }
  unsigned int m_till = m_since + m_range;
  
  std::cout << std::endl << "=============================================" << std::endl;
  std::cout << std::endl << "==================  STATUS  =================" << std::endl;
  std::cout << std::endl << "=============================================" << std::endl << std::endl;
  std::cout << ">> Range mode [" << m_since << ", " << m_till << "]" << std::endl;
  std::cout << std::endl << "=============================================" << std::endl << std::endl;
  
  RPCFw caen ( host, user, passw );
  std::vector<RPCObStatus::S_Item> Scheck;
  
  Scheck = caen.createSTATUS(m_since, m_till);
  Sdata = new RPCObStatus();
  RPCObStatus::S_Item Sfill;
  std::vector<RPCObStatus::S_Item>::iterator Sit;
  for(Sit = Scheck.begin(); Sit != Scheck.end(); Sit++){
    Sfill = *(Sit);
    Sdata->ObStatus_rpc.push_back(Sfill);
  }
  std::cout << " >> Final object size: " << Sdata->ObStatus_rpc.size() << std::endl;
   
  if (Sdata->ObStatus_rpc.size() == 0) {
    std::cout << "NO DATA TO BE STORED" << std::endl;
  }

  edm::TimeValue_t daqtime=0LL;
  daqtime=m_since;
  daqtime=(daqtime<<32);
   
  std::cout << "===> New IOV: since is = " << daqtime << std::endl;
  m_to_transfer.push_back(std::make_pair((RPCObStatus*)Sdata,daqtime));
  std::stringstream os;
  os<<"\n-->> NumberOfValue "<<Sdata->ObStatus_rpc.size()<<" until "<<m_till;
  m_userTextLog=os.str();
}

