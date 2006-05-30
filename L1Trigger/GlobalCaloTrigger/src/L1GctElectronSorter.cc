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
#include "FWCore/Utilities/interface/Exception.h"  

#include <iostream>

using namespace std;


L1GctElectronSorter::L1GctElectronSorter(int nInputs, bool iso, std::vector<L1GctSourceCard*> sourceCards):
  m_id(nInputs),
  m_emCandType(iso),
  m_theSCs(nInputs),
  m_inputCands(nInputs*4),
  m_outputCands(4)
{
  if(m_theSCs.size()!=sourceCards.size()){
    throw cms::Exception("L1GctSetupError")
      <<"L1GctElectronSorter::Constructor() : The number of Source Cards passed in the constructor doesn't correspond to the no of inputs given in the nInput variable";
  }
  
  for(unsigned i=0;i!=sourceCards.size();i++){
    if(sourceCards[i]!=0){
      m_theSCs[i] = sourceCards[i];
    }else{
      throw cms::Exception("L1GctSetupError")
	<<"L1GctElectronSorter::Constructor() : Pointer to Source Card #"<<i<<" is zero";
    }  
  }
  

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
  // loop over Source Cards - using integers not the vector size because the vector might be the wrong size
  for (unsigned int i=0; i<m_theSCs.size(); i++) {
    // loop over 4 candidates per Source Card
    for (unsigned int j=0; j<4; j++) {
      // get EM candidates, depending on type
      if (m_emCandType) {
	setInputEmCand((j+i*4),m_theSCs[i]->getIsoElectrons()[j]);
      }else {
	setInputEmCand((j+i*4),m_theSCs[i]->getNonIsoElectrons()[j]);
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

void L1GctElectronSorter::setInputEmCand(int i, L1GctEmCand cand){
  m_inputCands[i] = cand;
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



