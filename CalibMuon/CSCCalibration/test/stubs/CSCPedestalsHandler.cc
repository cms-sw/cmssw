#include "CalibMuon/CSCCalibration/test/stubs/CSCPedestalsHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include<iostream>

popcon::CSCDBPedestalsImpl::CSCDBPedestalsImpl(const edm::ParameterSet& pset): m_name(pset.getUntrackedParameter<std::string>("name","CSCDBPedestalsImpl")){}

popcon::CSCDBPedestalsImpl::~CSCDBPedestalsImpl(){}

void popcon::CSCDBPedestalsImpl::getNewObjects()
{

  std::cout << "------- CSC src - > getNewObjects\n"<<m_name;
  
  //check whats already inside of database
  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;
	
  unsigned int snc,tll;
  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;
  std::cout << "getNewObjects : enter till ? \n";
  std::cin >> tll;
  
  //the following code works, however since 1.6.0_pre7 it causes glibc 
  //double free error (inside CSC specific code) - commented 
  //
  //Using ES to get the data:

  edm::ESHandle<CSCDBPedestals> pedestal;
  mypedestals = pedestal.product();
  std::cout << "size " << pedestal->pedestals.size() << std::endl;
 
  /* 
  snc = edm::IOVSyncValue::beginOfTime().eventID().run();
  tll = edm::IOVSyncValue::endOfTime().eventID().run();//infinite
  */ 

  CSCDBPedestals * p1 = new CSCDBPedestals(*mypedestals);
  CSCDBPedestals * p2 = new CSCDBPedestals(*mypedestals);
  
  
  m_to_transfer.push_back(std::make_pair((CSCDBPedestals*)mypedestals,snc));
  m_to_transfer.push_back(std::make_pair((CSCDBPedestals*)p1,snc+20));
  m_to_transfer.push_back(std::make_pair((CSCDBPedestals*)p2,snc+10));
  
  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
}
