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


L1GctElectronSorter::L1GctElectronSorter(int id, bool iso):
  m_id(id),
  getIsoEmCands(iso),
  theSCs(5),
  inputCands(0),
  outputCands(4)
{
}

L1GctElectronSorter::~L1GctElectronSorter()
{

}

// clear buffers
void L1GctElectronSorter::reset() {
  inputCands.clear();
  outputCands.clear();	
  inputCands.clear();
  outputCands.clear();
}

// get the input data
void L1GctElectronSorter::fetchInput() {

  // loop over Source Cards - using integers not the vector size because the vector might be the wrong size
  for (unsigned int i=0; i<theSCs.size(); i++) {
    
    // loop over 4 candidates per Source Card
    for (unsigned int j=0; j<4; j++) {

      // get EM candidates, depending on type
      if (getIsoEmCands) {
	setInputEmCand(theSCs[i]->getIsoElectrons()[j]);
      }
      else {
	setInputEmCand(theSCs[i]->getNonIsoElectrons()[j]);
      }
    }
  }

}

//Process sorts the electron candidates after rank and stores the highest four (in the outputCands vector)
void L1GctElectronSorter::process() {

//Make temporary copy of data
    vector<L1GctEmCand> data = inputCands;
    
//Then sort it
    sort(data.begin(),data.end(),rank_gt());
  
//Copy data to output buffer
    for(int i = 0; i<4; i++){
      outputCands[i] = data[i];
    }
}

void L1GctElectronSorter::setInputSourceCard(unsigned int i, L1GctSourceCard* sc) {
  if (i < theSCs.size()) {
    theSCs[i]=sc;
  }
}

void L1GctElectronSorter::setInputEmCand(L1GctEmCand cand){
  inputCands.push_back(cand);
}

