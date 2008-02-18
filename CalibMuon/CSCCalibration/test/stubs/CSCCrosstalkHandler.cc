#include "CalibMuon/CSCCalibration/test/stubs/CSCCrosstalkHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include<iostream>

popcon::CSCDBCrosstalkImpl::CSCDBCrosstalkImpl(const edm::ParameterSet& pset): m_name(pset.getUntrackedParameter<std::string>("name","CSCDBCrosstalkImpl"))
{}

popcon::CSCDBCrosstalkImpl::~CSCDBCrosstalkImpl()
{
}


#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"


void popcon::CSCDBCrosstalkImpl::getNewObjects() {

  std::cout << "------- CSC src - > getNewObjects\n"<<m_name;
	

  // fill object from file
  CSCDBcrosstalk * cncrosstalk = CSCCrosstalkDBConditions::prefillDBCrosstalk();
  std::cout << "crosstalk size " << crosstalk->crosstalk.size() << std::endl;
 

  //check whats already inside of database
  
  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;  	

  unsigned int snc;
  
  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;


  
  m_to_transfer.push_back(std::make_pair(cncrosstalk,snc));
  
  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
}
