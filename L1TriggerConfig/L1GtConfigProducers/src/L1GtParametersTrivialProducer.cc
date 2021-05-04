/**
 * \class L1GtParametersTrivialProducer
 *
 *
 * Description: ESProducer for L1 GT parameters.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtParametersTrivialProducer.h"

// system include files
#include <memory>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"
#include <cstdint>

// forward declarations

// constructor(s)
L1GtParametersTrivialProducer::L1GtParametersTrivialProducer(const edm::ParameterSet& parSet) {
  // tell the framework what data is being produced
  setWhatProduced(this, &L1GtParametersTrivialProducer::produceGtParameters);

  // now do what ever other initialization is needed

  // total Bx's in the event

  m_totalBxInEvent = parSet.getParameter<int>("TotalBxInEvent");

  if (m_totalBxInEvent > 0) {
    if ((m_totalBxInEvent % 2) == 0) {
      m_totalBxInEvent = m_totalBxInEvent - 1;

      edm::LogInfo("L1GtParametersTrivialProducer")
          << "\nWARNING: Number of bunch crossing in event rounded to: " << m_totalBxInEvent
          << "\n         The number must be an odd number!\n"
          << std::endl;
    }
  } else {
    edm::LogInfo("L1GtParametersTrivialProducer")
        << "\nWARNING: Number of bunch crossing in event must be a positive number!"
        << "\n  Requested value was: " << m_totalBxInEvent << "\n  Reset to 1 (L1Accept bunch only).\n"
        << std::endl;

    m_totalBxInEvent = 1;
  }

  m_daqActiveBoards = static_cast<uint16_t>(parSet.getParameter<unsigned int>("DaqActiveBoards"));

  m_evmActiveBoards = static_cast<uint16_t>(parSet.getParameter<unsigned int>("EvmActiveBoards"));

  m_daqNrBxBoard = parSet.getParameter<std::vector<int> >("DaqNrBxBoard");

  m_evmNrBxBoard = parSet.getParameter<std::vector<int> >("EvmNrBxBoard");

  m_bstLengthBytes = parSet.getParameter<unsigned int>("BstLengthBytes");
}

// destructor
L1GtParametersTrivialProducer::~L1GtParametersTrivialProducer() {
  // empty
}

// member functions

// method called to produce the data
std::unique_ptr<L1GtParameters> L1GtParametersTrivialProducer::produceGtParameters(const L1GtParametersRcd& iRecord) {
  auto pL1GtParameters = std::make_unique<L1GtParameters>();

  // set total Bx's in the event
  pL1GtParameters->setGtTotalBxInEvent(m_totalBxInEvent);

  // set the active boards for L1 GT DAQ record
  pL1GtParameters->setGtDaqActiveBoards(m_daqActiveBoards);

  // set the active boards for L1 GT EVM record
  pL1GtParameters->setGtEvmActiveBoards(m_evmActiveBoards);

  // set the number of Bx per board for L1 GT DAQ record
  pL1GtParameters->setGtDaqNrBxBoard(m_daqNrBxBoard);

  // set the number of Bx per board for L1 GT EVM record
  pL1GtParameters->setGtEvmNrBxBoard(m_evmNrBxBoard);

  // set length of BST record (in bytes) for L1 GT EVM record
  pL1GtParameters->setGtBstLengthBytes(m_bstLengthBytes);

  return pL1GtParameters;
}
