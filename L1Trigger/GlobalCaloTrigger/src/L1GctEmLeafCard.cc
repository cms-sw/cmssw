#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmLeafCard.h"

#include <vector>
#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"

using std::vector;
using std::ostream;
using std::endl;

const unsigned L1GctEmLeafCard::N_SOURCE_CARDS = 9;
const unsigned L1GctEmLeafCard::N_SORTERS = 4;

L1GctEmLeafCard::L1GctEmLeafCard(int id, vector<L1GctSourceCard*> srcCards) :
  m_id(id),
  m_sorters(4),
  m_sourceCards(srcCards)
{

  // sorters 0 and 1 are in FPGA 0
  m_sorters[0] = new L1GctElectronSorter(true);
  m_sorters[1] = new L1GctElectronSorter(false);

  // sorters 2 and 3 are in FPGA 1
  m_sorters[2] = new L1GctElectronSorter(true);
  m_sorters[3] = new L1GctElectronSorter(false);
  
  // check for the right number of source cards
  if (m_sourceCards.size()!=N_SOURCE_CARDS) {
    throw cms::Exception("L1GctSetupError")
      << "L1GctEmLeafCard::L1GctEmLeafCard() : EM Leaf Card ID " << m_id << " has been incorrectly constructed!" << endl
      << "Expected " << N_SOURCE_CARDS << " source card pointers, only received " << m_sourceCards.size() << endl;
  }

  for (unsigned i=0; i<N_SOURCE_CARDS; i++) {
    if (m_sourceCards[i]==0) {
     throw cms::Exception("L1GctSetupError")
       << "L1GctEmLeafCard::L1GctEmLeafCard() : EM Leaf Card ID " << m_id << " has been incorrectly constructed!" << endl
       << "SourceCard pointer " << i << " is null" << endl;

    }
  }

}

L1GctEmLeafCard::~L1GctEmLeafCard() 
{
  delete m_sorters[0];
  delete m_sorters[1];
  delete m_sorters[2];
  delete m_sorters[3];
}


/// clear buffers
void L1GctEmLeafCard::reset() {
  for (unsigned i=0; i<N_SORTERS; i++) {
    m_sorters[i]->reset();
  }
}

/// fetch input data
void L1GctEmLeafCard::fetchInput() {
  for (unsigned i=0; i<N_SORTERS; i++) {
    m_sorters[i]->fetchInput();
  }
}

/// process the event
void L1GctEmLeafCard::process() {
  for (unsigned i=0; i<N_SORTERS; i++) {
    m_sorters[i]->process();
  }
}

/// get the output candidates
vector<L1GctEmCand> L1GctEmLeafCard::getOutputIsoEmCands(int fpga) {
  if (fpga<2) {
    return m_sorters[2*fpga]->OutputCands();
  }
  else {
    return vector<L1GctEmCand>(0);
  }
}

/// get the output candidates
vector<L1GctEmCand> L1GctEmLeafCard::getOutputNonIsoEmCands(int fpga) {
  if (fpga<2) {
    return m_sorters[2*fpga+1]->OutputCands();
  }
  else {
    return vector<L1GctEmCand>(0);
  }
}

ostream& operator<<(ostream& s, const L1GctEmLeafCard& card) {
  s << "No of Source Cards " <<card.m_sourceCards.size() << endl;
  s << "No of Electron Sorters " << card.m_sorters.size() << endl;
  s << endl;
  s << "Pointers in the Source Cards are: "<<endl;
  for (unsigned i=0; i<card.m_sourceCards.size(); i++) {
    if(i%6 == 0){
      s << endl;
    }
    s << card.m_sourceCards[i]<<"  ";
  }
  s << endl;
  s <<"Pointers in the sorters vector are: " << endl;
  for (unsigned i=0; i<card.m_sorters.size(); i++) {
    s << card.m_sorters[i]<<"  ";
  }
  s << endl;
  s << "Other private members (objects and variables): "<<endl;
  s << "Card (algorithm?) ID "<<card.m_id<<endl;
  return s;
}
