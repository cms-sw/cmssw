#include "CalibMuon/CSCCalibration/test/stubs/CSCFakeDBCrosstalkHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <iostream>

#include "CalibMuon/CSCCalibration/interface/CSCFakeDBCrosstalk.h"
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"

popcon::CSCFakeDBCrosstalkImpl::CSCFakeDBCrosstalkImpl(const edm::ParameterSet &pset)
    : m_name(pset.getUntrackedParameter<std::string>("name", "CSCFakeDBCrosstalkImpl")) {}

popcon::CSCFakeDBCrosstalkImpl::~CSCFakeDBCrosstalkImpl() {}

void popcon::CSCFakeDBCrosstalkImpl::getNewObjects() {
  std::cout << "------- CSC src - > getNewObjects\n" << m_name;

  // fill object from file
  CSCDBCrosstalk *cncrosstalk = CSCFakeDBCrosstalk::prefillDBCrosstalk();
  // std::cout << "crosstalk size " << cncrosstalk->crosstalk.size() <<
  // std::endl;

  // check whats already inside of database

  std::cerr << "got offlineInfo" << std::endl;
  std::cerr << tagInfo().name << " , last object valid since " << tagInfo().lastInterval.first << std::endl;

  unsigned int snc;

  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;

  m_to_transfer.push_back(std::make_pair(cncrosstalk, snc));

  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
}
