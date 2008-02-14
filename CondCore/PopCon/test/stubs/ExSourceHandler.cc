#include "ExSourceHandler.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<iostream>


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
  m_name(pset.getUntrackedParameter<std::string>("name","ExPedestalSource")){
}

popcon::ExPedestalSource::~ExPedestalSource()
{
 
}

void popcon::ExPedestalSource::getNewObjects() {
   std::cerr << "------- " << m_name 
	     << " - > getNewObjects" << std::endl;
  //check whats already inside of database
  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << ", size " << tagInfo().size 
            << " last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;  
  if (tagInfo().size>0) {
    Ref payload = lastPayload();
    std::cerr<<"size of last payload  "<< 
      payload->m_pedestals.size()<<std::endl;
  }

  
  
  
  unsigned int snc;
  
  std::cerr << "Source implementation test ::getNewObjects : enter first since ? \n";
  std::cin >> snc;



  
  Pedestals * p0 = new Pedestals;
  fill(*p0,3);
  Pedestals * p1 = new Pedestals;
  fill(*p1,5);
  Pedestals * p2 = new Pedestals;
  fill(*p2,7);
  m_to_transfer.push_back(std::make_pair((Pedestals*)p0,snc));
  m_to_transfer.push_back(std::make_pair((Pedestals*)p1,snc+20));
  m_to_transfer.push_back(std::make_pair((Pedestals*)p2,snc+10));

  std::cerr << "------- " << m_name << " - > getNewObjects" << std::endl;
}
