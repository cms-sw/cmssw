#include "CalibMuon/CSCCalibration/test/stubs/CSCBadStripsHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include<iostream>

#include "CondFormats/CSCObjects/interface/CSCBadStrips.h"
#include "CalibMuon/CSCCalibration/interface/CSCBadStripsConditions.h"

popcon::CSCBadStripsImpl::CSCBadStripsImpl(const edm::ParameterSet& pset): m_name(pset.getUntrackedParameter<std::string>("name","CSCBadStripsImpl"))
{}

popcon::CSCBadStripsImpl::~CSCBadStripsImpl()
{
}

void popcon::CSCBadStripsImpl::getNewObjects()
{

  std::cout << "------- CSC src - > getNewObjects\n"<<m_name;
  
  // fill object from file
  CSCBadStrips * cnbadstrips = CSCBadStripsConditions::prefillBadStrips();
  
  //check whats already inside of database
  
  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl; 
  
  unsigned int snc;
  
  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;
 
  
  m_to_transfer.push_back(std::make_pair(cnbadstrips,snc));
  
  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n"<< std::endl;
}
