/**
 * \class L1GtBoardMapsTester
 *
 *
 * Description: test analyzer for various mappings of the L1 GT boards.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtBoardMapsTester.h"

// system include files
#include <iomanip>
#include <iostream>

// user include files
//   base class

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
#include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"

// forward declarations

// constructor(s)
L1GtBoardMapsTester::L1GtBoardMapsTester(const edm::ParameterSet& parSet) : m_getToken(esConsumes()) {
  // empty
}

// loop over events
void L1GtBoardMapsTester::analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup& evSetup) const {
  L1GtBoardMaps const& l1GtBM = evSetup.getData(m_getToken);

  l1GtBM.print(std::cout);
  std::cout << std::endl;

  // print for simplicity the individual maps

  l1GtBM.printGtDaqRecordMap(std::cout);
  std::cout << std::endl;

  l1GtBM.printGtEvmRecordMap(std::cout);
  std::cout << std::endl;

  l1GtBM.printGtDaqActiveBoardsMap(std::cout);
  std::cout << std::endl;

  l1GtBM.printGtEvmActiveBoardsMap(std::cout);
  std::cout << std::endl;

  l1GtBM.printGtBoardSlotMap(std::cout);
  std::cout << std::endl;

  l1GtBM.printGtBoardHexNameMap(std::cout);
  std::cout << std::endl;

  l1GtBM.printGtQuadToPsbMap(std::cout);
  std::cout << std::endl;
}
