#include "OnlineDB/CSCCondDB/test/stubs/CSCChamberIndexHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <iostream>

#include "CondFormats/CSCObjects/interface/CSCChamberIndex.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberIndexValues.h"

popcon::CSCChamberIndexImpl::CSCChamberIndexImpl(const edm::ParameterSet& pset)
    : m_name(pset.getUntrackedParameter<std::string>("name", "CSCChamberIndexImpl")) {}

popcon::CSCChamberIndexImpl::~CSCChamberIndexImpl() {}

void popcon::CSCChamberIndexImpl::getNewObjects() {
  std::cout << "------- CSC src - > getNewObjects\n" << m_name;

  // fill object from file
  CSCChamberIndex* mycham_index = CSCChamberIndexValues::fillChamberIndex();

  //check whats already inside of database
  std::cerr << "got offlineInfo" << std::endl;
  std::cerr << tagInfo().name << " , last object valid since " << tagInfo().lastInterval.first << std::endl;

  unsigned int snc;

  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;

  m_to_transfer.push_back(std::make_pair((CSCChamberIndex*)mycham_index, snc));

  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
}
