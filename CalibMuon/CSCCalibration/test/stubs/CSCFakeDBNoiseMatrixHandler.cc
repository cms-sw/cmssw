#include "CalibMuon/CSCCalibration/test/stubs/CSCFakeDBNoiseMatrixHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <iostream>

#include "CalibMuon/CSCCalibration/interface/CSCFakeDBNoiseMatrix.h"
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"

popcon::CSCFakeDBNoiseMatrixImpl::CSCFakeDBNoiseMatrixImpl(const edm::ParameterSet &pset)
    : m_name(pset.getUntrackedParameter<std::string>("name", "CSCFakeDBNoiseMatrixImpl")) {}

popcon::CSCFakeDBNoiseMatrixImpl::~CSCFakeDBNoiseMatrixImpl() {}

void popcon::CSCFakeDBNoiseMatrixImpl::getNewObjects() {
  std::cout << "------- CSC src - > getNewObjects\n" << m_name;

  // fill object from file
  CSCDBNoiseMatrix *cnmatrix = CSCFakeDBNoiseMatrix::prefillDBNoiseMatrix();
  // std::cout << "matrix size " << cnmatrix->matrix.size() << std::endl;

  // check whats already inside of database

  std::cerr << "got offlineInfo" << std::endl;
  std::cerr << tagInfo().name << " , last object valid since " << tagInfo().lastInterval.first << std::endl;

  unsigned int snc;

  std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
  std::cin >> snc;

  m_to_transfer.push_back(std::make_pair(cnmatrix, snc));

  std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n" << std::endl;
}
