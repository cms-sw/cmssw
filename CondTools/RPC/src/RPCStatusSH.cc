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
  m_test_suptype(pset.getUntrackedParameter<int>("supType",0)),
  m_since(pset.getUntrackedParameter<unsigned long long>("since",1)),
  m_range(pset.getUntrackedParameter<unsigned long long>("range",72000)),
  n_IOVmax(pset.getUntrackedParameter<int>("n_iovmax",1))
{
}

popcon::RpcDataS::~RpcDataS()
{
}

void popcon::RpcDataS::getNewObjects() {
  
  unsigned long long  presentTime = time(NULL);
  std::cout << "------- " << m_name << " - > getNewObjects\n" 
	    << "got offlineInfo "<< tagInfo().name 
	    << ", size " << tagInfo().size 
	    << ", last object valid since "<< tagInfo().lastInterval.first 
	    << " token "<< tagInfo().lastPayloadToken << std::endl;
  
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

  unsigned long long m_till = m_since + m_range;
  int n_iov=0;
  RPCFw caen ( host, user, passw );
  caen.setSuptype(m_test_suptype);
  std::map<int,RPCObStatus::S_Item> lastStatus;

  // Let's take the last payload from DB and take the last value of each detid
  if (tagInfo().size>0){
    std::vector<RPCObStatus::S_Item> pSt = lastPayload()->ObStatus_rpc;
    std::cout <<"Number of payload of the lastIOV "<<pSt.size()<<std::endl;
    std::vector<RPCObStatus::S_Item>::iterator iS;
    for (iS=pSt.begin();iS<pSt.end();iS++){
      if (lastStatus.find(iS->detid) != lastStatus.end()){
        if (lastStatus.find(iS->detid)->second.unixtime < iS->unixtime){
          lastStatus[iS->detid]=*iS;
        }
      }else{
        lastStatus[iS->detid]=*iS;
      }
    }
    std::cout <<"Number of lastValues for lastIOV "<<lastStatus.size()<<std::endl;
  }

  std::cout <<"m_till==== "<<m_till<<std::endl;
  while((presentTime - m_till)>m_range/2 && n_iov < n_IOVmax ){
    
    //Fill the last value of each Sudet in the payload updating unixtime to since...
    Sdata = new RPCObStatus();
    for (std::map<int, RPCObStatus::S_Item>::iterator l=lastStatus.begin(); l!=lastStatus.end(); l++){
      RPCObStatus::S_Item slast=l->second;
      slast.unixtime=m_since;
      Sdata->ObStatus_rpc.push_back(slast);
    }
    std::cout <<"Starting with "<<Sdata->ObStatus_rpc.size()<<" monitoring points from previous IOV"<<std::endl;
    
    std::cout << std::endl << "=============================================" << std::endl;
    std::cout << std::endl << "==================  STATUS  =================" << std::endl;
    std::cout << std::endl << "=============================================" << std::endl << std::endl;
    std::cout << ">> Range mode [" << m_since << ", " << m_till << "]" << std::endl;
    std::cout << std::endl << "=============================================" << std::endl << std::endl;
    
    std::vector<RPCObStatus::S_Item> Scheck;
    
    Scheck = caen.createSTATUS(m_since, m_till);
    RPCObStatus::S_Item Sfill;
    std::vector<RPCObStatus::S_Item>::iterator Sit;
    for(Sit = Scheck.begin(); Sit != Scheck.end(); Sit++){
      Sfill = *(Sit);
      if (Sit->unixtime < m_since || Sit->unixtime > m_till){
	std::cout <<"Wrong time  since("<<m_since<<")-"<<Sit->unixtime<<"-("<<m_till<<")"<<std::endl;
      }
      Sdata->ObStatus_rpc.push_back(Sfill);
      if (lastStatus.find(Sit->detid) != lastStatus.end()){
	if (lastStatus.find(Sit->detid)->second.unixtime < Sit->unixtime){
	  lastStatus[Sit->detid]=Sfill;
	}
      }else{
	lastStatus[Sit->detid]=Sfill;
      }
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

    //  delete Sdata;                                                                                          
    n_iov++;
    m_since=m_till;
    m_till=m_since+m_range;
  }
}

