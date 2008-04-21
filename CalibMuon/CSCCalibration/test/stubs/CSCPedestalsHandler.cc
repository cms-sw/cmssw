#include "CalibMuon/CSCCalibration/test/stubs/CSCPedestalsHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include<iostream>

#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CalibMuon/CSCCalibration/interface/CSCPedestalsDBConditions.h"

popcon::CSCDBPedestalsImpl::CSCDBPedestalsImpl(const edm::ParameterSet& pset): m_name(pset.getUntrackedParameter<std::string>("name","CSCDBPedestalsImpl")){}

popcon::CSCDBPedestalsImpl::~CSCDBPedestalsImpl(){}

void popcon::CSCDBPedestalsImpl::getNewObjects()
{

  std::cout << "CSCPedestalsHandler - time before filling object:"<< std::endl;
  int id=system("date");
  std::cout << "------- CSC src - > getNewObjects\n"<<m_name;
  
  // fill object from file
  CSCDBPedestals * cnpedestals = CSCPedestalsDBConditions::prefillDBPedestals();
  //std::cout << "pedestals size " << cnpedestals->pedestals.size() << std::endl;

  std::cout << "CSCPedestalsHandler - time after filling object:"<< std::endl;
  id=system("date");

  //check whats already inside of database

  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;
	
  unsigned int snc;
  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;
  std::cout << "getNewObjects : enter till ? \n";
  
   
  id=system("date");
  m_to_transfer.push_back(std::make_pair(cnpedestals,snc));
   
  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
  std::cout << "CSCPedestalsHandler - time before writing into DB:"<< std::endl;
  id=system("date");
}
