#include "CalibMuon/CSCCalibration/test/stubs/CSCCrosstalkHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include<iostream>

popcon::CSCDBCrosstalkImpl::CSCDBCrosstalkImpl(const edm::ParameterSet& pset): m_name(pset.getUntrackedParameter<std::string>("name","CSCDBCrosstalkImpl"))
{}

popcon::CSCDBCrosstalkImpl::~CSCDBCrosstalkImpl()
{
}

void popcon::CSCDBCrosstalkImpl::getNewObjects()
{

  std::cout << "------- CSC src - > getNewObjects\n"<<m_name;
	
  //check whats already inside of database
  
  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;  	

  unsigned int snc;
  
  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;
  //std::cout << "getNewObjects : enter till ? \n";
  //std::cin >> tll;
  
  
  //the following code works, however since 1.6.0_pre7 it causes glibc 
  //double free error (inside CSC specific code) - commented 
  //
  //Using ES to get the data:
  /*
  edm::ESHandle<CSCDBCrosstalk> crosstalk;
  mycrosstalk = crosstalk.product();
  std::cout << "size " << crosstalk->crosstalk.size() << std::endl;
  */

  CSCDBCrosstalk * p0 = new CSCDBCrosstalk;
  CSCDBCrosstalk * p1 = new CSCDBCrosstalk;
  CSCDBCrosstalk * p2 = new CSCDBCrosstalk;
  
  m_to_transfer.push_back(std::make_pair((CSCDBCrosstalk*)p0,snc));
  m_to_transfer.push_back(std::make_pair((CSCDBCrosstalk*)p1,snc+20));
  m_to_transfer.push_back(std::make_pair((CSCDBCrosstalk*)p2,snc+10));
  
  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
}
