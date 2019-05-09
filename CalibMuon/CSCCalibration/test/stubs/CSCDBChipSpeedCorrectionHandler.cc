#include "CalibMuon/CSCCalibration/test/stubs/CSCDBChipSpeedCorrectionHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <iostream>

#include "CalibMuon/CSCCalibration/interface/CSCChipSpeedCorrectionDBConditions.h"
#include "CondFormats/CSCObjects/interface/CSCDBChipSpeedCorrection.h"

popcon::CSCDBChipSpeedCorrectionImpl::CSCDBChipSpeedCorrectionImpl(const edm::ParameterSet &pset) {
  m_name = (pset.getUntrackedParameter<std::string>("name", "CSCDBChipSpeedCorrectionImpl"));
  isForMC = (pset.getUntrackedParameter<bool>("isForMC", true));
  dataCorrFileName = (pset.getUntrackedParameter<std::string>("dataCorrFileName", "empty.txt"));
  dataOffset = 170.;
}

popcon::CSCDBChipSpeedCorrectionImpl::~CSCDBChipSpeedCorrectionImpl() {}

void popcon::CSCDBChipSpeedCorrectionImpl::getNewObjects() {
  std::cout << "CSCChipSpeedCorrectionHandler - time before filling object:" << std::endl;
  std::cout << "------- CSC src - > getNewObjects\n" << m_name;

  // fill object from file
  // bool isForMC = iConfig.getUntrackedParameter<bool>("isForMC",true);
  // string dataCorrFileName=
  // iConfig.getUntrackedParameter<std::string>("dataCorrFileName","empty.txt");
  CSCDBChipSpeedCorrection *cnchipspeed =
      CSCChipSpeedCorrectionDBConditions::prefillDBChipSpeedCorrection(isForMC, dataCorrFileName, dataOffset);
  // std::cout << "chipspeed size " << cnchipspeed->chipspeed.size() <<
  // std::endl;

  std::cout << "CSCChipSpeedCorrectionHandler - time after filling object:" << std::endl;

  // check whats already inside of database

  std::cerr << "got offlineInfo" << std::endl;
  std::cerr << tagInfo().name << " , last object valid since " << tagInfo().lastInterval.first << std::endl;

  unsigned int snc;
  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;
  std::cout << "getNewObjects : enter till ? \n";

  m_to_transfer.push_back(std::make_pair(cnchipspeed, snc));

  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
  std::cout << "CSCChipSpeedCorrectionHandler - time before writing into DB:" << std::endl;
}
