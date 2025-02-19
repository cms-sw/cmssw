#include "OnlineDB/CSCCondDB/test/stubs/CSCCrateMapHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include<iostream>

#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
#include "OnlineDB/CSCCondDB/interface/CSCCrateMapValues.h"

popcon::CSCCrateMapImpl::CSCCrateMapImpl(const edm::ParameterSet& pset): m_name(pset.getUntrackedParameter<std::string>("name","CSCCrateMapImpl"))
{}

popcon::CSCCrateMapImpl::~CSCCrateMapImpl()
{
}

void popcon::CSCCrateMapImpl::getNewObjects()
{

  std::cout << "------- CSC src - > getNewObjects\n"<<m_name;
  
  // fill object from file
  CSCCrateMap * mycrate_map = CSCCrateMapValues::fillCrateMap();
  
  //check whats already inside of database  
  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl; 
  
  unsigned int snc;
  
  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;
 
  
  m_to_transfer.push_back(std::make_pair((CSCCrateMap*)mycrate_map,snc));
  
  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n"<< std::endl;
}
