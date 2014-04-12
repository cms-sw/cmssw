#include "CalibMuon/CSCCalibration/test/stubs/CSCL1TPParametersHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include<iostream>

popcon::CSCL1TPParametersImpl::CSCL1TPParametersImpl(const edm::ParameterSet& pset): m_name(pset.getUntrackedParameter<std::string>("name","CSCL1TPParametersImpl"))
{}

popcon::CSCL1TPParametersImpl::~CSCL1TPParametersImpl()
{
}


#include "CondFormats/CSCObjects/interface/CSCL1TPParameters.h"
#include "CalibMuon/CSCCalibration/interface/CSCL1TPParametersConditions.h"



void popcon::CSCL1TPParametersImpl::getNewObjects() {

  std::cout << "CSCL1TPParametersHandler - time before filling object:"<< std::endl;
  std::cout << "------- CSC src - > getNewObjects\n"<<m_name;
	

  // fill object from file
  CSCL1TPParameters * cnl1tp = CSCL1TPParametersConditions::prefillCSCL1TPParameters();
 
  std::cout << "CSCL1TPParametersHandler - time after filling object:"<< std::endl;

  //check whats already inside of database
  
  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;  	

  unsigned int snc;
  
  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;

  
  m_to_transfer.push_back(std::make_pair(cnl1tp,snc));
  
  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
  std::cout << "CSCL1TPParametersHandler - time before writing into DB:"<< std::endl;
}
