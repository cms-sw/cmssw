#include "OnlineDB/CSCCondDB/test/stubs/CSCDDUMapHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <iostream>

#include "CondFormats/CSCObjects/interface/CSCDDUMap.h"
#include "OnlineDB/CSCCondDB/interface/CSCDDUMapValues.h"

popcon::CSCDDUMapImpl::CSCDDUMapImpl(const edm::ParameterSet& pset)
    : m_name(pset.getUntrackedParameter<std::string>("name", "CSCDDUMapImpl")) {}

popcon::CSCDDUMapImpl::~CSCDDUMapImpl() {}

void popcon::CSCDDUMapImpl::getNewObjects() {
  std::cout << "------- CSC src - > getNewObjects\n" << m_name;

  // fill object from file
  CSCDDUMap* myddu_map = CSCDDUMapValues::fillDDUMap();

  //check whats already inside of database
  std::cerr << "got offlineInfo" << std::endl;
  std::cerr << tagInfo().name << " , last object valid since " << tagInfo().lastInterval.first << std::endl;

  unsigned int snc;

  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;

  m_to_transfer.push_back(std::make_pair((CSCDDUMap*)myddu_map, snc));

  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
}
