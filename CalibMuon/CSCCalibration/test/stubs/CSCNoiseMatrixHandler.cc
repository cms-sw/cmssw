#include "CalibMuon/CSCCalibration/test/stubs/CSCNoiseMatrixHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include<iostream>

popcon::CSCDBNoiseMatrixImpl::CSCDBNoiseMatrixImpl(const edm::ParameterSet& pset): m_name(pset.getUntrackedParameter<std::string>("name","CSCDBNoiseMatrixImpl")){}

popcon::CSCDBNoiseMatrixImpl::~CSCDBNoiseMatrixImpl(){}

void popcon::CSCDBNoiseMatrixImpl::getNewObjects()
{

  std::cout << "------- CSC src - > getNewObjects\n"<<m_name;
  
  //check whats already inside of database
  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;  
  
  unsigned int snc,tll;
  
  std::cerr << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;
  std::cerr << "getNewObjects : enter till ? \n";
  std::cin >> tll;
  
  
  //the following code works, however since 1.6.0_pre7 it causes glibc 
  //double free error (inside CSC specific code) - commented 
  //
  //Using ES to get the data:
  
  edm::ESHandle<CSCDBNoiseMatrix> matrix;
  mymatrix = matrix.product();
  std::cout << "size " << matrix->matrix.size() << std::endl;
  
  //changed to an empty object
  //mymatrix = new CSCDBNoiseMatrix();
  
  CSCDBNoiseMatrix * p1 = new CSCDBNoiseMatrix(*mymatrix);
  CSCDBNoiseMatrix * p2 = new CSCDBNoiseMatrix(*mymatrix);
  
  m_to_transfer.push_back(std::make_pair((CSCDBNoiseMatrix*)mymatrix,snc));
  m_to_transfer.push_back(std::make_pair((CSCDBNoiseMatrix*)p1,snc+20));
  m_to_transfer.push_back(std::make_pair((CSCDBNoiseMatrix*)p2,snc+10));
  
  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n"<< std::endl;
}
