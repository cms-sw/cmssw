#include "CalibMuon/CSCCalibration/test/stubs/CSCBadChambersHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <iostream>

#include "CalibMuon/CSCCalibration/interface/CSCBadChambersConditions.h"
#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"

popcon::CSCBadChambersImpl::CSCBadChambersImpl(const edm::ParameterSet &pset)
    : m_name(pset.getUntrackedParameter<std::string>("name", "CSCBadChambersImpl")) {}

popcon::CSCBadChambersImpl::~CSCBadChambersImpl() {}

void popcon::CSCBadChambersImpl::getNewObjects() {
  std::cout << "------- CSC src - > getNewObjects\n" << m_name;

  // fill object from file
  CSCBadChambers *cnbadchambers = CSCBadChambersConditions::prefillBadChambers();

  // check whats already inside of database

  std::cerr << "got offlineInfo" << std::endl;
  std::cerr << tagInfo().name << " , last object valid since " << tagInfo().lastInterval.first << std::endl;

  unsigned int snc;

  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;

  m_to_transfer.push_back(std::make_pair(cnbadchambers, snc));

  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
}
