#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmLeafCard.h"

#include <vector>

L1GctEmLeafCard::L1GctEmLeafCard(int id) :
  m_id(id),
  m_sorters(4),
  m_sourceCards(9)
{
}

L1GctEmLeafCard::~L1GctEmLeafCard() {
}


/// clear buffers
void L1GctEmLeafCard::reset() {
  for (unsigned i=0; i<m_sorters.size(); i++) {
    m_sorters[i]->reset();
  }
}

/// fetch input data
void L1GctEmLeafCard::fetchInput() {
  for (unsigned i=0; i<m_sorters.size(); i++) {
    m_sorters[i]->fetchInput();
  }
}

/// process the event
void L1GctEmLeafCard::process() {
  for (unsigned i=0; i<m_sorters.size(); i++) {
    m_sorters[i]->process();
  }
}

/// add a source card as input
void L1GctEmLeafCard::setInputSourceCard(int i, L1GctSourceCard* sc) {
  if (i < m_sourceCards.size()) {
    m_sourceCards[i]=sc;
  }
}

/// get the output candidates
vector<L1GctEmCand> L1GctEmLeafCard::getOutputIsoEmCands() {
   return m_sorters[0]->getOutput();
}

/// get the output candidates
vector<L1GctEmCand> L1GctEmLeafCard::getOutputNonIsoEmCands() {
     return m_sorters[1]->getOutput();
}
