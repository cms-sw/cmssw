
/*! \file L1GctElectronFinalSort.cc
 * \Class that does the final sorting of electron candidates
 *
 * This class sorts the electron candidates by rank in 
 * ascending order. Inputs are the 4 highest Et electrons from
 * the leaf? cards
 *
 * \author  Maria Hansen
 * \date    12/05/06
 * \version 1.2
 */

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronFinalSort.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmLeafCard.h"


L1GctElectronFinalSort::L1GctElectronFinalSort(bool iso):
  m_emCandsType(iso),
  m_theLeafCards(2),
  m_inputCands(8),
  m_outputCands(4)
{
}

L1GctElectronFinalSort::~L1GctElectronFinalSort(){
  m_inputCands.clear();
  m_outputCands.clear();
}

void L1GctElectronFinalSort::reset(){
  m_inputCands.clear();
  m_outputCands.clear();
}

void L1GctElectronFinalSort::fetchInput() {

  for (int i=0; i<2; i++) { /// loop over leaf cards
    for (int j=0; j<2; j++) { /// loop over FPGAs
      for (int k=0; k<4; k++) {  /// loop over candidates
	if (m_emCandsType) {
	  std::vector<L1GctEmCand> isoCands;
	  isoCands = m_theLeafCards[i]->getOutputIsoEmCands(j);
	  setInputEmCand((i*4)+(j*2)+k, isoCands[k]);
	}
	else {
	  std::vector<L1GctEmCand> nonIsoCands;
	  //	  setInputEmCand((i*4)+(j*2)+k, m_theLeafCards[i]->getOutputNonIsoEmCands(j)[k]);
	  nonIsoCands = m_theLeafCards[i]->getOutputNonIsoEmCands(j);
	  setInputEmCand((i*4)+(j*2)+k, nonIsoCands[k]);
	}
      }
    }   
  }

}

void L1GctElectronFinalSort::process(){
//Make temporary copy of data
    std::vector<L1GctEmCand> data = m_inputCands;
    
//Then sort it
    sort(data.begin(),data.end(),rank_gt());
  
//Copy data to output buffer
    for(int i = 0; i<4; i++){
     m_outputCands[i] = data[i];
    }
}

void L1GctElectronFinalSort::setInputLeafCard(int i, L1GctEmLeafCard* card) {
  if (i<2) {
    m_theLeafCards[i] = card;
  }
}

void L1GctElectronFinalSort::setInputEmCand(int i, L1GctEmCand cand){
  m_inputCands[i] = cand;
}

std::ostream& operator<<(std::ostream& s, const L1GctElectronFinalSort& cand) {
  s << "No of Electron Leaf Cards " << cand.m_theLeafCards.size() << std::endl;
  s << "No of Electron Input Candidates " << cand.m_inputCands.size() << std::endl;
  s << "No of Electron Output Candidates" << cand.m_outputCands.size() << std::endl;
  s << std::endl;
  s << "Pointers in the Electron Leaf cards are: "<<std::endl;
  for (unsigned i=0; i<cand.m_theLeafCards.size(); i++) {
    s << cand.m_theLeafCards[i]<<"  ";
  }
  s << std::endl;
  s <<"Pointers in the Input Candidates vector are: " << std::endl;
  for (unsigned i=0; i<cand.m_inputCands.size(); i++) {
    s << cand.m_inputCands[i]<<"  ";
  }
  s << std::endl;
  s << "Pointers in the Output Candidates vector are: "<<std::endl;
  for (unsigned i=0; i<cand.m_outputCands.size(); i++) {
    s << cand.m_outputCands[i]<<"  ";
  }
  s << std::endl;
  s << "Other private members (objects and variables): "<<std::endl;
  s << "Type of electron card is "<<cand.m_emCandsType<<std::endl;
  return s;
}


