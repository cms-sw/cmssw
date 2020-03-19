#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetSorter.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

L1GctJetSorter::L1GctJetSorter() : m_inputJets() {}
L1GctJetSorter::L1GctJetSorter(L1GctJetSorter::JetVector& inputJets) : m_inputJets(inputJets) {}

L1GctJetSorter::~L1GctJetSorter() {}

void L1GctJetSorter::setJets(L1GctJetSorter::JetVector& inputJets) { m_inputJets = inputJets; }

L1GctJetSorter::JetVector L1GctJetSorter::getSortedJets() const {
  unsigned nJets = m_inputJets.size();
  std::vector<unsigned> position(nJets, 0);
  // Replicate the firmware jet sorting algorithm.
  // If two jets in the input array have equal rank,
  // the one that occurs first in the array has higher priority.
  for (unsigned j1 = 0; j1 < nJets; j1++) {
    for (unsigned j2 = j1 + 1; j2 < nJets; j2++) {
      if (m_inputJets.at(j1).rank() < m_inputJets.at(j2).rank()) {
        position.at(j1) = position.at(j1) + 1;
      } else {
        position.at(j2) = position.at(j2) + 1;
      }
    }
  }

  JetVector result(m_inputJets.size());
  for (unsigned j1 = 0; j1 < nJets; j1++) {
    result.at(position.at(j1)) = m_inputJets.at(j1);
  }

  return result;
}

L1GctJetSorter::JetVector L1GctJetSorter::getInputJets() const { return m_inputJets; }
