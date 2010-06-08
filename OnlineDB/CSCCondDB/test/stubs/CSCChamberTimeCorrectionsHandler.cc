#include "OnlineDB/CSCCondDB/test/stubs/CSCChamberTimeCorrectionsHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include<iostream>

#include "CondFormats/CSCObjects/interface/CSCChamberTimeCorrections.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberTimeCorrectionsValues.h"

popcon::CSCChamberTimeCorrectionsImpl::CSCChamberTimeCorrectionsImpl(const edm::ParameterSet& pset){
  m_name= (pset.getUntrackedParameter<std::string>("name","CSCChamberTimeCorrectionsImpl"));
  isForMC=(pset.getUntrackedParameter<bool>("isForMC",true));
}

popcon::CSCChamberTimeCorrectionsImpl::~CSCChamberTimeCorrectionsImpl()
{
}

void popcon::CSCChamberTimeCorrectionsImpl::getNewObjects()
{

  std::cout << "------- CSC src - > getNewObjects\n"<<m_name;
  
  // fill object from file
  CSCChamberTimeCorrections * mychambers = CSCChamberTimeCorrectionsValues::prefill(isForMC);
  
  //check whats already inside of database  
  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl; 
  
  unsigned int snc;
  
  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;
 
  
  m_to_transfer.push_back(std::make_pair((CSCChamberTimeCorrections*)mychambers,snc));
  
  std::cout << "-------  " << m_name << "CSC src - > getNewObjects -----------\n"<< std::endl;
}
