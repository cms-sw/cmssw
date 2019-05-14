#include "CalibMuon/CSCCalibration/test/stubs/CSCBadWiresHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <iostream>

#include "CalibMuon/CSCCalibration/interface/CSCBadWiresConditions.h"
#include "CondFormats/CSCObjects/interface/CSCBadWires.h"

popcon::CSCBadWiresImpl::CSCBadWiresImpl(const edm::ParameterSet &pset)
    : m_name(pset.getUntrackedParameter<std::string>("name", "CSCBadWiresImpl")) {}

popcon::CSCBadWiresImpl::~CSCBadWiresImpl() {}

void popcon::CSCBadWiresImpl::getNewObjects() {
  std::cout << "------- CSC src - > getNewObjects\n" << m_name;

  // fill object from file
  CSCBadWires *cnbadwires = CSCBadWiresConditions::prefillBadWires();

  // check whats already inside of database

  std::cerr << "got offlineInfo" << std::endl;
  std::cerr << tagInfo().name << " , last object valid since " << tagInfo().lastInterval.first << std::endl;

  unsigned int snc;

  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;

  m_to_transfer.push_back(std::make_pair(cnbadwires, snc));

  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
}
