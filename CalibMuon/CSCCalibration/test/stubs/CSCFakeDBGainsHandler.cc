#include "CalibMuon/CSCCalibration/test/stubs/CSCFakeDBGainsHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include<iostream>

#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBGains.h"

popcon::CSCFakeDBGainsImpl::CSCFakeDBGainsImpl(const edm::ParameterSet& pset): m_name(pset.getUntrackedParameter<std::string>("name","CSCFakeDBGainsImpl"))
{}

popcon::CSCFakeDBGainsImpl::~CSCFakeDBGainsImpl()
{
}

void popcon::CSCFakeDBGainsImpl::getNewObjects()
{

  std::cout << "------- CSC src - > getNewObjects\n"<<m_name;
  
  // fill object from file
  CSCDBGains * cngains = CSCFakeDBGains::prefillDBGains();
  //std::cout << "gains size " << cngains->gains.size() << std::endl;
  
  //check whats already inside of database
  
  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl; 
  
  unsigned int snc;
  
  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;
 
  
  m_to_transfer.push_back(std::make_pair(cngains,snc));
  
  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n"<< std::endl;
}
