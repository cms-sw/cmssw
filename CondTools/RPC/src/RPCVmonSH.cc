/*
 *  See headers for a description
 *
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "CondTools/RPC/interface/RPCVmonSH.h"
#include<sstream>
#include<iostream>
#include <ctime>

popcon::RpcDataV::RpcDataV(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","RpcData")),
  host(pset.getUntrackedParameter<std::string>("host", "source db host")),
  user(pset.getUntrackedParameter<std::string>("user", "source username")),
  passw(pset.getUntrackedParameter<std::string>("passw", "source password")),
  m_first(pset.getUntrackedParameter<bool>("first",false)),
  m_test_suptype(pset.getUntrackedParameter<int>("supType",0)),
  m_since(pset.getUntrackedParameter<unsigned long long>("since",1)),
  m_range(pset.getUntrackedParameter<unsigned long long>("range",72000)),
  n_IOVmax(pset.getUntrackedParameter<int>("n_iovmax",1))
{
}

popcon::RpcDataV::~RpcDataV()
{
}

void popcon::RpcDataV::getNewObjects() {

  unsigned long long  presentTime = time(NULL);
  std::cout << "------- " << m_name << " - > getNewObjects\n" 
	    << "got offlineInfo "<< tagInfo().name 
	    << ", size " << tagInfo().size 
	    << ", last object valid since " << tagInfo().lastInterval.first 
	    << " token "   << tagInfo().lastPayloadToken << std::endl;

  std::cout << logDBEntry().usertext << "last record with the correct tag has been written in the db: "
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
  int n_iov=0;
  RPCFw caen ( host, user, passw );
  caen.setSuptype(m_test_suptype);
  std::map<int,RPCObVmon::V_Item> lastVmon;

 // Let's take the last payload from DB and take the last value of each detid
  if (tagInfo().size>0){
    std::vector<RPCObVmon::V_Item> pVm = lastPayload()->ObVmon_rpc;
    std::cout <<"Number of payload of the lastIOV "<<pVm.size()<<std::endl;
    std::vector<RPCObVmon::V_Item>::iterator iV;
    for (iV=pVm.begin();iV<pVm.end();iV++){
      if (lastVmon.find(iV->detid) != lastVmon.end()){
        if (lastVmon.find(iV->detid)->second.unixtime < iV->unixtime){
          lastVmon[iV->detid]=*iV;
        }
      }else{
        lastVmon[iV->detid]=*iV;
      }
    }
    std::cout <<"Number of lastValues for lastIOV "<<lastVmon.size()<<std::endl;
  }

  std::cout <<"m_till==== "<<m_till<<std::endl;
  while((presentTime - m_till)>m_range/2 && n_iov < n_IOVmax ){

    //Fill the last value of each Sudet in the payload updating unixtime to since...
    Vdata = new RPCObVmon();
    for (std::map<int, RPCObVmon::V_Item>::iterator l=lastVmon.begin(); l!=lastVmon.end(); l++){
      RPCObVmon::V_Item vlast=l->second;
      vlast.unixtime=m_since;
      Vdata->ObVmon_rpc.push_back(vlast);
    }
    std::cout <<"Starting with "<<Vdata->ObVmon_rpc.size()<<" monitoring points from previous IOV"<<std::endl;
    
    std::cout << std::endl << "=============================================" << std::endl;
    std::cout << std::endl << "===================  VMON  ==================" << std::endl;
    std::cout << std::endl << "=============================================" << std::endl << std::endl;
    std::cout << ">> Range mode [" << m_since << ", " << m_till << "]" << std::endl;
    std::cout << std::endl << "=============================================" << std::endl << std::endl;   
    
    std::vector<RPCObVmon::V_Item> Vcheck;  

    Vcheck = caen.createVMON(m_since, m_till);
    RPCObVmon::V_Item Vfill;
    std::vector<RPCObVmon::V_Item>::iterator Vit;
    for(Vit = Vcheck.begin(); Vit != Vcheck.end(); Vit++){
      Vfill = *(Vit);
      if (Vit->unixtime < m_since || Vit->unixtime > m_till){
	std::cout <<"Wrong time  since("<<m_since<<")-"<<Vit->unixtime<<"-("<<m_till<<")"<<std::endl;
      }
      Vdata->ObVmon_rpc.push_back(Vfill);
      if (lastVmon.find(Vit->detid) != lastVmon.end()){
        if (lastVmon.find(Vit->detid)->second.unixtime < Vit->unixtime){
          lastVmon[Vit->detid]=Vfill;
        }
      }else{
        lastVmon[Vit->detid]=Vfill;
      }
    }
    std::cout << " >> Final object size: " << Vdata->ObVmon_rpc.size() << std::endl;
   
    if (Vdata->ObVmon_rpc.size() == 0) {
      std::cout << "NO DATA TO BE STORED" << std::endl;
    }
   
    edm::TimeValue_t daqtime=0LL;
    daqtime=m_since;
    daqtime=(daqtime<<32);

    std::cout << "===> New IOV: since is = " << daqtime << std::endl;
    m_to_transfer.push_back(std::make_pair((RPCObVmon*)Vdata,daqtime));
    std::stringstream os;
    os<<"\n-->> NumberOfValue "<<Vdata->ObVmon_rpc.size()<<" until "<<m_till;
    m_userTextLog=os.str();
    
    n_iov++;
    m_since=m_till;
    m_till=m_since+m_range;
  }
}

