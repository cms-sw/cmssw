
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
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>


L1GctElectronFinalSort::L1GctElectronFinalSort(bool iso, L1GctEmLeafCard* card1,L1GctEmLeafCard* card2):
  m_emCandsType(iso),
  m_theLeafCards(2),
  m_inputCands(8),
  m_outputCands(4)
{
  if(card1!=0){
    m_theLeafCards[0] = card1;
  }else{
    throw cms::Exception("L1GctSetupError")
      <<"L1GctElectronFinalSort::Constructor() : 1st EmLeafCard passed is zero";
      }
  if(card2!=0){
    m_theLeafCards[1] = card2;
  }else{
    throw cms::Exception("L1GctSetupError")
      <<"L1GctElectronFinalSort::Constructor() : 2nd EmLeafCard passed is zero";
  }
}

L1GctElectronFinalSort::~L1GctElectronFinalSort(){
  m_inputCands.clear();
  m_outputCands.clear();
}

void L1GctElectronFinalSort::reset(){
  for(unsigned i = 0;i!=m_inputCands.size();i++){    
    m_inputCands[i] = 0;
  }
  for(unsigned i = 0;i!=m_outputCands.size();i++){  
    m_outputCands[i] = 0;
  }
}

void L1GctElectronFinalSort::fetchInput() {
  for (int i=0; i<2; i++) { /// loop over leaf cards
    for (int j=0; j<2; j++) { /// loop over FPGAs
      for (int k=0; k<4; k++) {  /// loop over candidates
	if (m_emCandsType) {
	  setInputEmCand((i*4)+(j*2)+k, m_theLeafCards[i]->getOutputIsoEmCands(j)[k]); 
	}
	else {
	  setInputEmCand((i*4)+(j*2)+k, m_theLeafCards[i]->getOutputNonIsoEmCands(j)[k]);
	}
      }
    }   
  }  
}

void L1GctElectronFinalSort::process(){
  //First check that there are electron candidates in the input vector
  for(unsigned i=0;i!=m_inputCands.size();i++){
    if(m_inputCands[i].rank()==0){
      throw cms::Exception("L1GctSetupError")
	<<"L1GctElectronFinalSort::process() : No input candidate in vector at "<<i<<" ";
    }else{
   
      //Make temporary copy of data
      std::vector<L1GctEmCand> data = m_inputCands;
      
      //Then sort it
      sort(data.begin(),data.end(),rank_gt());
      
      //Copy data to output buffer
      for(int i = 0; i<4; i++){
	m_outputCands[i] = data[i];
      }
    }
  }
}

void L1GctElectronFinalSort::setInputEmCand(int i, L1GctEmCand cand){
  m_inputCands[i] = cand;
}

std::ostream& operator<<(std::ostream& s, const L1GctElectronFinalSort& cand) {
  s << "===ElectronFinalSort===" << std::endl;
  s << "Card type = " <<cand.m_emCandsType<<std::endl;
  s << "No of Electron Leaf Cards " << cand.m_theLeafCards.size() << std::endl;
  s << "Pointers to the Electron Leaf cards are: "<<std::endl;
  for (unsigned i=0; i<cand.m_theLeafCards.size(); i++) {
    s << cand.m_theLeafCards[i]<<"  ";
  }
  s << std::endl;
  s << "No of Electron Input Candidates " << cand.m_inputCands.size() << std::endl;
  s << "No of Electron Output Candidates " << cand.m_outputCands.size() << std::endl;
   
  return s;
}


