#include "CalibMuon/CSCCalibration/test/stubs/CSCDBGasGainCorrectionHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include<iostream>

#include "CondFormats/CSCObjects/interface/CSCDBGasGainCorrection.h"
#include "CalibMuon/CSCCalibration/interface/CSCGasGainCorrectionDBConditions.h"

popcon::CSCDBGasGainCorrectionImpl::CSCDBGasGainCorrectionImpl(const edm::ParameterSet& pset){
  m_name=(pset.getUntrackedParameter<std::string>("name","CSCDBGasGainCorrectionImpl"));
  isForMC=(pset.getUntrackedParameter<bool>("isForMC",true));
  dataCorrFileName=(pset.getUntrackedParameter<std::string>("dataCorrFileName","empty.txt"));
}

popcon::CSCDBGasGainCorrectionImpl::~CSCDBGasGainCorrectionImpl(){}

void popcon::CSCDBGasGainCorrectionImpl::getNewObjects()
{

  std::cout << "CSCGasGainCorrectionHandler - time before filling object:"<< std::endl;
  std::cout << "------- CSC src - > getNewObjects\n"<<m_name;
  
  // fill object from file
  //bool isForMC = iConfig.getUntrackedParameter<bool>("isForMC",true);
  //string dataCorrFileName= iConfig.getUntrackedParameter<std::string>("dataCorrFileName","empty.txt");
  CSCDBGasGainCorrection * cngasgain = CSCGasGainCorrectionDBConditions::prefillDBGasGainCorrection(isForMC,dataCorrFileName);

  std::cout << "CSCGasGainCorrectionHandler - time after filling object:"<< std::endl;

  //check whats already inside of database

  std::cerr<<"got offlineInfo"<<std::endl;
  std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;
	
  unsigned int snc;
  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;
  std::cout << "getNewObjects : enter till ? \n";
  
   
  m_to_transfer.push_back(std::make_pair(cngasgain,snc));
   
  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
  std::cout << "CSCGasGainCorrectionHandler - time before writing into DB:"<< std::endl;
}
