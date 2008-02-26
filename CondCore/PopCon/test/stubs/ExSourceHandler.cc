#include "ExSourceHandler.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include<iostream>
#include<vector>

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
  m_since(pset.getUntrackedParameter<unsigned int >("firstSince",5)),
  m_number(pset.getUntrackedParameter<unsigned int >("number",3)){
}

popcon::ExPedestalSource::~ExPedestalSource()
{
 
}

void popcon::ExPedestalSource::getNewObjects() {
   edm::LogInfo   ("ExPedestalsSource") << "------- " << m_name 
	     << " - > getNewObjects\n" << 
  //check whats already inside of database
      "got offlineInfo"<<
    tagInfo().name << ", size " << tagInfo().size 
            << ", last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;  
  

  if (tagInfo().size>0) {
    Ref payload = lastPayload();
    edm::LogInfo   ("ExPedestalsSource")<<"size of last payload  "<< 
      payload->m_pedestals.size()<<std::endl;
  }

  
  std::cout<<"first since = "<< m_since <<std::endl;
  
 
  
  Pedestals * p0 = new Pedestals;
  fill(*p0,3);
  m_to_transfer.push_back(std::make_pair((Pedestals*)p0,m_since));
  
  unsigned int since = m_since+10*m_number;
  unsigned int size = 5;
  for (int j=1; j<(int)m_number;++j) {
    Pedestals * p1 = new Pedestals;
    fill(*p1,size);
    m_to_transfer.push_back(std::make_pair((Pedestals*)p1,since));
    since-=10;
    size+=2;
  }

  edm::LogInfo   ("ExPedestalsSource") << "------- " << m_name << " - > getNewObjects" << std::endl;
}
