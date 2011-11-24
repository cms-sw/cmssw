/*
 *  See headers for a description
 *
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "CondTools/RPC/interface/RPCImonSH.h"
#include <sstream>
#include <iostream>

popcon::RpcDataI::RpcDataI(const edm::ParameterSet& pset) :
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

popcon::RpcDataI::~RpcDataI()
{
}

void popcon::RpcDataI::getNewObjects() {

  unsigned long long  presentTime = time(NULL);
  std::cout << "------- " << m_name << " - > getNewObjects\n" 
	    << "got offlineInfo "<< tagInfo().name 
	    << ", size " << tagInfo().size 
	    << ", last object valid since "<< tagInfo().lastInterval.first 
	    << " token " << tagInfo().lastPayloadToken << std::endl;

  std::cout << logDBEntry().usertext << "\nlast record with the correct tag has been written in the db: "
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

  unsigned long long m_till = m_since + m_range;
  int n_iov=0;
  RPCFw caen ( host, user, passw );
  caen.setSuptype(m_test_suptype);
  std::map<int,RPCObImon::I_Item> lastImon;
 
  // Let's take the last payload from DB and take the last value of each detid
  if (tagInfo().size>0){
    std::vector<RPCObImon::I_Item> pIm = lastPayload()->ObImon_rpc;
    std::cout <<"Number of payload of the lastIOV "<<pIm.size()<<std::endl;
    std::vector<RPCObImon::I_Item>::iterator iI;
    for (iI=pIm.begin();iI<pIm.end();iI++){
      if (lastImon.find(iI->detid) != lastImon.end()){
	if (lastImon.find(iI->detid)->second.unixtime < iI->unixtime){
	  lastImon[iI->detid]=*iI;
	}
      }else{
	lastImon[iI->detid]=*iI;
      }
    }
    std::cout <<"Number of lastValues for lastIOV "<<lastImon.size()<<std::endl;
  }
  
  std::cout <<"m_till==== "<<m_till<<std::endl;
  while((presentTime - m_till)>m_range/2 && n_iov < n_IOVmax ){
    
    //Fill the last value of each Sudet in the payload updating unixtime to since...
    Idata = new RPCObImon();
    for (std::map<int, RPCObImon::I_Item>::iterator l=lastImon.begin(); l!=lastImon.end(); l++){
      RPCObImon::I_Item ilast=l->second;
      ilast.unixtime=m_since;
      Idata->ObImon_rpc.push_back(ilast);
    }
    std::cout <<"Starting with "<<Idata->ObImon_rpc.size()<<" monitoring points from previous IOV"<<std::endl;

    std::cout << std::endl << "=============================================" << std::endl;
    std::cout << std::endl << "===================  IMON  ==================" << std::endl;
    std::cout << std::endl << "=============================================" << std::endl << std::endl;
    std::cout << ">> Range mode [" << m_since << ", " << m_till << "]" << std::endl;
    std::cout << std::endl << "=============================================" << std::endl << std::endl;

    std::vector<RPCObImon::I_Item> Icheck;
  
    Icheck = caen.createIMON(m_since, m_till);
    RPCObImon::I_Item Ifill;
    std::vector<RPCObImon::I_Item>::iterator Iit;
    for(Iit = Icheck.begin(); Iit != Icheck.end(); Iit++){
      Ifill = *(Iit);
      if (Iit->unixtime < m_since || Iit->unixtime > m_till){
	std::cout <<"Wrong time  since("<<m_since<<")-"<<Iit->unixtime<<"-("<<m_till<<")"<<std::endl;
      }
      Idata->ObImon_rpc.push_back(Ifill);
      if (lastImon.find(Iit->detid) != lastImon.end()){
        if (lastImon.find(Iit->detid)->second.unixtime < Iit->unixtime){
          lastImon[Iit->detid]=Ifill;
	}
      }else{
        lastImon[Iit->detid]=Ifill;
      }
    }
    std::cout << ">> Final object size: " << Idata->ObImon_rpc.size() << std::endl;
    
    if (Idata->ObImon_rpc.size()== 0) {
      std::cout << "NO DATA TO BE STORED" << std::endl;
    }
    
    edm::TimeValue_t daqtime=0LL;
    daqtime=m_since;
    daqtime=(daqtime<<32);
  
    std::cout << "===> New IOV: since is = " << daqtime << std::endl;
    m_to_transfer.push_back(std::make_pair((RPCObImon*)Idata,daqtime));
    std::stringstream os;
    os<<"\n-->> NumberOfValue "<<Idata->ObImon_rpc.size()<<" until "<<m_till;
    m_userTextLog=os.str();
  
    //  delete Idata;
    n_iov++;
    m_since=m_till;				
    m_till=m_since+m_range;
  }
}

