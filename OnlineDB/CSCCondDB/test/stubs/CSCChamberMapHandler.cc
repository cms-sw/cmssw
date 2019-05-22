#include "OnlineDB/CSCCondDB/test/stubs/CSCChamberMapHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <iostream>

#include "CondFormats/CSCObjects/interface/CSCChamberMap.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberMapValues.h"

popcon::CSCChamberMapImpl::CSCChamberMapImpl(const edm::ParameterSet& pset)
    : m_name(pset.getUntrackedParameter<std::string>("name", "CSCChamberMapImpl")) {}

popcon::CSCChamberMapImpl::~CSCChamberMapImpl() {}

void popcon::CSCChamberMapImpl::getNewObjects() {
  std::cout << "------- CSC src - > getNewObjects\n" << m_name;

  // fill object from file
  CSCChamberMap* mycham_map = CSCChamberMapValues::fillChamberMap();

  //check whats already inside of database
  std::cerr << "got offlineInfo" << std::endl;
  std::cerr << tagInfo().name << " , last object valid since " << tagInfo().lastInterval.first << std::endl;

  unsigned int snc;

  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;

  m_to_transfer.push_back(std::make_pair((CSCChamberMap*)mycham_map, snc));

  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
}
