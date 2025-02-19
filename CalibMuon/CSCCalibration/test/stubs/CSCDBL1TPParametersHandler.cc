#include "CalibMuon/CSCCalibration/test/stubs/CSCDBL1TPParametersHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include<iostream>

popcon::CSCDBL1TPParametersImpl::CSCDBL1TPParametersImpl(const edm::ParameterSet& pset): m_name(pset.getUntrackedParameter<std::string>("name","CSCDBL1TPParametersImpl"))
{}

popcon::CSCDBL1TPParametersImpl::~CSCDBL1TPParametersImpl()
{
}


#include "CondFormats/CSCObjects/interface/CSCDBL1TPParameters.h"
#include "CalibMuon/CSCCalibration/interface/CSCDBL1TPParametersConditions.h"



void popcon::CSCDBL1TPParametersImpl::getNewObjects() {

  std::cout << "CSCDBL1TPParametersHandler - time before filling object:"<< std::endl;
  std::cout << "------- CSC src - > getNewObjects\n"<<m_name;
	

  // fill object from file
  CSCDBL1TPParameters * cnl1tp = CSCDBL1TPParametersConditions::prefillCSCDBL1TPParameters();
 
  std::cout << "CSCDBL1TPParametersHandler - time after filling object:"<< std::endl;

  //check whats already inside of database
  
  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;  	

  unsigned int snc;
  
  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;

  
  m_to_transfer.push_back(std::make_pair(cnl1tp,snc));
  
  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
  std::cout << "CSCDBL1TPParametersHandler - time before writing into DB:"<< std::endl;
}
