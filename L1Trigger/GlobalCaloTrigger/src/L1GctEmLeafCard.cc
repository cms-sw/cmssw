#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmLeafCard.h"
#include <vector>

using std::endl;
using std::ostream;
using std::vector;

const unsigned L1GctEmLeafCard::N_SORTERS = 4;

L1GctEmLeafCard::L1GctEmLeafCard(int id) : m_id(id), m_sorters(4) {
  // sorters 0 and 1 are in FPGA U1 and deal with RCT crates 4-8 (13-17)
  m_sorters.at(0) = new L1GctElectronSorter(5, true);
  m_sorters.at(1) = new L1GctElectronSorter(5, false);

  // sorters 2 and 3 are in FPGA U2 and deal with RCT crates 0-3 (9-12)
  m_sorters.at(2) = new L1GctElectronSorter(4, true);
  m_sorters.at(3) = new L1GctElectronSorter(4, false);
}

L1GctEmLeafCard::~L1GctEmLeafCard() {
  delete m_sorters.at(0);
  delete m_sorters.at(1);
  delete m_sorters.at(2);
  delete m_sorters.at(3);
}

/// clear buffers
void L1GctEmLeafCard::reset() {
  L1GctProcessor::reset();
  for (unsigned i = 0; i < N_SORTERS; i++) {
    m_sorters.at(i)->reset();
  }
}

/// partially clear buffers
void L1GctEmLeafCard::setBxRange(const int firstBx, const int numberOfBx) {
  L1GctProcessor::setBxRange(firstBx, numberOfBx);
  for (unsigned i = 0; i < N_SORTERS; i++) {
    m_sorters.at(i)->setBxRange(firstBx, numberOfBx);
  }
}

/// partially clear buffers
void L1GctEmLeafCard::setNextBx(const int bx) {
  L1GctProcessor::setNextBx(bx);
  for (unsigned i = 0; i < N_SORTERS; i++) {
    m_sorters.at(i)->setNextBx(bx);
  }
}

/// fetch input data
void L1GctEmLeafCard::fetchInput() {
  for (unsigned i = 0; i < N_SORTERS; i++) {
    m_sorters.at(i)->fetchInput();
  }
}

/// process the event
void L1GctEmLeafCard::process() {
  for (unsigned i = 0; i < N_SORTERS; i++) {
    m_sorters.at(i)->process();
  }
}

/// get the output candidates
vector<L1GctEmCand> L1GctEmLeafCard::getOutputIsoEmCands(int fpga) {
  if (fpga < 2) {
    return m_sorters.at(2 * fpga)->getOutputCands();
  } else {
    return vector<L1GctEmCand>(0);
  }
}

/// get the output candidates
vector<L1GctEmCand> L1GctEmLeafCard::getOutputNonIsoEmCands(int fpga) {
  if (fpga < 2) {
    return m_sorters.at(2 * fpga + 1)->getOutputCands();
  } else {
    return vector<L1GctEmCand>(0);
  }
}

ostream& operator<<(ostream& s, const L1GctEmLeafCard& card) {
  s << "===L1GctEmLeafCard===" << endl;
  s << "ID = " << card.m_id << endl;
  s << "No of Electron Sorters = " << card.m_sorters.size() << endl;
  for (unsigned i = 0; i < card.m_sorters.size(); i++) {
    s << std::endl;
    s << "===ElectronSorter===" << std::endl;
    s << "ElectronSorter no: " << i << endl << (*card.m_sorters.at(i));
  }
  s << endl;
  return s;
}
