#include "ExSourceHandler.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


//#include<iostream>
#include<sstream>
#include<vector>
#include<string>
#include <sstream>


namespace {

  void fill(Pedestals & p, int nc) {
    p.m_pedestals.reserve(nc);
    for(int ichannel=1; ichannel<=nc; ++ichannel){
      Pedestals::Item item;
      item.m_mean=1.11*ichannel;
      item.m_variance=1.12*ichannel;
      p.m_pedestals.push_back(item);
    }
  }
}

popcon::ExPedestalSource::ExPedestalSource(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","ExPedestalSource")),
  m_since(pset.getUntrackedParameter<long long>("firstSince",5)),
  m_increment(pset.getUntrackedParameter<long long>("increment",10)),
  m_number(pset.getUntrackedParameter<long long>("number",3)){
}

popcon::ExPedestalSource::~ExPedestalSource()
{
 
}

void popcon::ExPedestalSource::getNewObjects() {
  edm::LogInfo ("ExPedestalsSource") 
    << "------- " << m_name 
    << " - > getNewObjects\n"
    //check whats already inside of database
    << "got offlineInfo"
    << tagInfo().name << ", size " << tagInfo().size    
    << ", last object valid since " 
    << tagInfo().lastInterval.first << " token "   
    << tagInfo().lastPayloadToken << std::endl;

  //edm::LogInfo ("ExPedestalsSource")
  //  << " ------ last entry info regarding the payload (if existing): " <<logDBEntry().usertext 
  //  << "; last record with the correct tag (if existing) has been written in the db: " 
  //  <<logDBEntry().destinationDB<< std::endl; 
  
  if (tagInfo().size>0) {
    Ref payload = lastPayload();
    edm::LogInfo   ("ExPedestalsSource")<<"size of last payload  "<< 
      payload->m_pedestals.size()<<std::endl;
    
  }

 
  std::cout<<"first since = "<< m_since <<std::endl;
  
  
  Pedestals * p0 = new Pedestals;
  fill(*p0,3);
  m_to_transfer.push_back(std::make_pair((Pedestals*)p0, m_since));
  
  unsigned long long since = (unsigned long long)(m_since+m_increment*(m_number-1));
  
  std::cout<<"last since = "<< since <<std::endl;
  
  unsigned long long size = 5;
  for (unsigned long long j=1; j<(unsigned long long)m_number;++j) {
    Pedestals * p1 = new Pedestals;
    fill(*p1,size);
    m_to_transfer.push_back(std::make_pair((Pedestals*)p1,since));
    since-=m_increment;
    size+=2;
  }

  std::ostringstream ss; 
  ss << "num=" << m_number <<","
     << "firstSize=3," <<"lastSize=" << size-2 << std::endl;

    
  std::ostringstream fsince;
  fsince<< "iov fisrtsince == " << m_since; 

  m_userTextLog = ss.str()+ ";" + fsince.str();
 
  
  edm::LogInfo   ("ExPedestalsSource") << "------- " << m_name << " - > getNewObjects" << std::endl;
}
