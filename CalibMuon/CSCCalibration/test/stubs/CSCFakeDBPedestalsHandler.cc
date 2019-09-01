#include "CalibMuon/CSCCalibration/test/stubs/CSCFakeDBPedestalsHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <iostream>

#include "CalibMuon/CSCCalibration/interface/CSCFakeDBPedestals.h"
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"

popcon::CSCFakeDBPedestalsImpl::CSCFakeDBPedestalsImpl(const edm::ParameterSet &pset)
    : m_name(pset.getUntrackedParameter<std::string>("name", "CSCFakeDBPedestalsImpl")) {}

popcon::CSCFakeDBPedestalsImpl::~CSCFakeDBPedestalsImpl() {}

void popcon::CSCFakeDBPedestalsImpl::getNewObjects() {
  std::cout << "------- CSC src - > getNewObjects\n" << m_name;

  // fill object from file
  CSCDBPedestals *cnpedestals = CSCFakeDBPedestals::prefillDBPedestals();
  // std::cout << "peds size " << cnpedestals->pedestals.size() << std::endl;

  // check whats already inside of database

  std::cerr << "got offlineInfo" << std::endl;
  std::cerr << tagInfo().name << " , last object valid since " << tagInfo().lastInterval.first << std::endl;

  unsigned int snc;

  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;

  m_to_transfer.push_back(std::make_pair(cnpedestals, snc));

  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
}
