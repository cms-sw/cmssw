/*! \file L1GctElectronSorter.cc
 * \Class that sort electron candidates
 *
 * This class sorts the electron candidates by rank in 
 * ascending order.
 *
 * \author  Maria Hansen
 * \date    21/04/06
 * \version 1.1
 */

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"

#include <iostream>

using namespace std;

L1GctElectronSorter::L1GctElectronSorter(int id, bool iso):
  m_id(id),
  m_emCandType(iso),
  m_theSCs(5),
  m_inputCands(0),
  m_outputCands(4)
{
}

L1GctElectronSorter::~L1GctElectronSorter()
{

}

// clear buffers
void L1GctElectronSorter::reset() {
  m_inputCands.clear();
  m_outputCands.clear();	
}

// get the input data
void L1GctElectronSorter::fetchInput() {
  ///REMEMBER TO REMOVE
  cout<<"In electron sorter fetchInput method, SC's "<<m_theSCs.size()<<endl;
  // loop over Source Cards - using integers not the vector size because the vector might be the wrong size
  for (unsigned int i=0; i<m_theSCs.size(); i++) {
    
    // loop over 4 candidates per Source Card
    for (unsigned int j=0; j<4; j++) {

      // get EM candidates, depending on type
      if (m_emCandType) {
	cout <<"card type is "<<m_emCandType<<endl;
	//	cout<<"values in the private source card vector
	setInputEmCand(m_theSCs[i]->getIsoElectrons()[j]);
	//	vector<L1GctEmCand> data = m_theSCs[i]->getIsoElectrons();
	//for(unsigned m=0;m!=data.size();m++){
	//  setInputEmCand(data[m]);
	//}
	}
      else {
	setInputEmCand(m_theSCs[i]->getNonIsoElectrons()[j]);
      }
    }
  }

}

//Process sorts the electron candidates after rank and stores the highest four (in the outputCands vector)
void L1GctElectronSorter::process() {

//Make temporary copy of data
    std::vector<L1GctEmCand> data = m_inputCands;
    
//Then sort it
    sort(data.begin(),data.end(),rank_gt());
  
//Copy data to output buffer
    for(int i = 0; i<4; i++){
      m_outputCands[i] = data[i];
    }
}

void L1GctElectronSorter::setInputSourceCard(unsigned int i, L1GctSourceCard* sc) {
  if (i < m_theSCs.size()) {
    m_theSCs[i]=sc;
  }
}

void L1GctElectronSorter::setInputEmCand(L1GctEmCand cand){
  m_inputCands.push_back(cand);
}

std::ostream& operator<<(std::ostream& s, const L1GctElectronSorter& cand) {
  s << "No of Source Cards " << cand.m_theSCs.size() << std::endl;
  s << "No of Electron Input Candidates " << cand.m_inputCands.size()<< std::endl;
  s << "No of Electron Output Candidates" << cand.m_outputCands.size()<< std::endl;
  s << "Pointers in the Source Cards are: "<<std::endl;
  for (unsigned i=0; i<cand.m_theSCs.size(); i++) {
    s <<cand.m_theSCs[i]<<"  ";
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
  s << "Type of electron card is "<<cand.m_emCandType<<std::endl;
  s << "Algorithm ID "<<cand.m_id<<std::endl;
  return s;
}



