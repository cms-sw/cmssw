#include "ExSourceHandler.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<iostream>

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
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;  

  
  
  
  unsigned int snc;
  
  std::cerr << "Source implementation test ::getNewObjects : enter first since ? \n";
  std::cin >> snc;



  
  Pedestals * p0 = new Pedestals;
  Pedestals * p1 = new Pedestals;
  Pedestals * p2 = new Pedestals;
  m_to_transfer.push_back(std::make_pair((Pedestals*)p0,snc));
  m_to_transfer.push_back(std::make_pair((Pedestals*)p1,snc+20));
  m_to_transfer.push_back(std::make_pair((Pedestals*)p2,snc+10));

  std::cerr << "------- " << m_name << " - > getNewObjects" << std::endl;
}
