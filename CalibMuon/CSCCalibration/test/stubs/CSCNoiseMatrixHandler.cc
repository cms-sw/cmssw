#include "CalibMuon/CSCCalibration/test/stubs/CSCNoiseMatrixHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include<iostream>

#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CalibMuon/CSCCalibration/interface/CSCNoiseMatrixDBConditions.h"

popcon::CSCDBNoiseMatrixImpl::CSCDBNoiseMatrixImpl(const edm::ParameterSet& pset): m_name(pset.getUntrackedParameter<std::string>("name","CSCDBNoiseMatrixImpl")){}

popcon::CSCDBNoiseMatrixImpl::~CSCDBNoiseMatrixImpl(){}

void popcon::CSCDBNoiseMatrixImpl::getNewObjects()
{

  std::cout << "CSCNoiseMatrixHandler - time before filling object:"<< std::endl;
  std::cout << "------- CSC src - > getNewObjects\n"<<m_name;
  
  // fill object from file
  CSCDBNoiseMatrix * cndbmatrix = CSCNoiseMatrixDBConditions::prefillDBNoiseMatrix();
  //std::cout << "crosstalk size " << cndbmatrix->matrix.size() << std::endl;
  
  std::cout << "CSCNoiseMatrixHandler - time after filling object:"<< std::endl;

  //check whats already inside of database

  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;  
  
  unsigned int snc;
  
  std::cerr << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;
    
  m_to_transfer.push_back(std::make_pair(cndbmatrix,snc));
  
  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n"<< std::endl;
  std::cout << "CSCNoiseMatrixHandler - time before writing into DB:"<< std::endl;
}
