#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmLeafCard.h"

#include <vector>
#include <iostream>

using namespace std;

L1GctEmLeafCard::L1GctEmLeafCard(int id) :
  m_id(id),
  m_sorters(4),
  m_sourceCards(9)
{
  m_sorters[0] = new L1GctElectronSorter(true);
  m_sorters[1] = new L1GctElectronSorter(true);
  m_sorters[2] = new L1GctElectronSorter(false);
  m_sorters[3] = new L1GctElectronSorter(false);
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
  cout<<"In EmLeafCard in fetchInput method "<<m_sorters.size()<<endl;
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
  if ( i < m_sourceCards.size()) {
    m_sourceCards[i]=sc;
  }
  //REMOVE
  //  m_sorters[i]->setInputSourceCard(i, sc); 
}

/// get the output candidates
std::vector<L1GctEmCand> L1GctEmLeafCard::getOutputIsoEmCands(int fpga) {
   return m_sorters[fpga]->OutputCands();
}

/// get the output candidates
std::vector<L1GctEmCand> L1GctEmLeafCard::getOutputNonIsoEmCands(int fpga) {
     return m_sorters[fpga+2]->OutputCands();
}

std::ostream& operator<<(std::ostream& s, const L1GctEmLeafCard& card) {
  s << "No of Source Cards " <<card.m_sourceCards.size() << std::endl;
  s << "No of Electron Sorters " << card.m_sorters.size() << std::endl;
  s << std::endl;
  s << "Pointers in the Source Cards are: "<<std::endl;
  for (unsigned i=0; i<card.m_sourceCards.size(); i++) {
    if(i%6 == 0){
      s << std::endl;
    }
    s << card.m_sourceCards[i]<<"  ";
  }
  s << std::endl;
  s <<"Pointers in the sorters vector are: " << std::endl;
  for (unsigned i=0; i<card.m_sorters.size(); i++) {
    s << card.m_sorters[i]<<"  ";
  }
  s << std::endl;
  s << "Other private members (objects and variables): "<<std::endl;
  s << "Card (algorithm?) ID "<<card.m_id<<std::endl;
   return s;
}
