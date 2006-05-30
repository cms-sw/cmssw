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
  vector<L1GctSourceCard*> firstHalf;
  vector<L1GctSourceCard*> secondHalf;
  
  
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

  for(unsigned i=0;i!=m_sourceCards.size();i++){
    if(i<4){
      firstHalf.push_back(m_sourceCards[i]);
    }else{
      secondHalf.push_back(m_sourceCards[i]);
    }
  }

  // sorters 0 and 1 are in FPGA 0
  m_sorters[0] = new L1GctElectronSorter(4,true, firstHalf);
  m_sorters[1] = new L1GctElectronSorter(4,false,firstHalf);
  
  // sorters 2 and 3 are in FPGA 1
  m_sorters[2] = new L1GctElectronSorter(5,true, secondHalf);
  m_sorters[3] = new L1GctElectronSorter(5,false,secondHalf);
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
    return m_sorters[2*fpga]->getOutputCands();
  }
  else {
    return vector<L1GctEmCand>(0);
  }
}

/// get the output candidates
vector<L1GctEmCand> L1GctEmLeafCard::getOutputNonIsoEmCands(int fpga) {
  if (fpga<2) {
    return m_sorters[2*fpga+1]->getOutputCands();
  }
  else {
    return vector<L1GctEmCand>(0);
  }
}

ostream& operator<<(ostream& s, const L1GctEmLeafCard& card) {
  s << "No of Source Cards " <<card.m_sourceCards.size() << endl;
  s << "No of Electron Sorters " << card.m_sorters.size() << endl;
  s << endl;
  s << "Pointers to L1GctSourceCard : "<<endl;
  for (unsigned i=0; i<card.m_sourceCards.size(); i++) {
    s << i << " " << card.m_sourceCards[i] << endl; //" id : " << card.m_sourceCards[i]->id() << endl;
  }
  s << endl;
  s <<"Pointers to L1GctElectronSorter : " << endl;
  for (unsigned i=0; i<card.m_sorters.size(); i++) {
    s << card.m_sorters[i]<<"  ";
  }
  s << endl;
  s << "Other private members (objects and variables): "<<endl;
  s << "Card ID "<<card.m_id<<endl;
  return s;
}
