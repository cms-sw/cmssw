
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

#include <iostream>

using std::cout;



L1GctElectronFinalSort::L1GctElectronFinalSort(bool iso):
  getIsoEmCands(iso),
  theLeafCards(2),
  inputCands(8),
  outputCands(4)
{
}

L1GctElectronFinalSort::~L1GctElectronFinalSort(){
}

void L1GctElectronFinalSort::reset(){
  inputCands.clear();
  outputCands.clear();
}

void L1GctElectronFinalSort::fetchInput() {

  for (int i=0; i<2; i++) { /// loop over leaf cards
    for (int j=0; j<2; j++) { /// loop over FPGAs
      for (int k=0; k<4; k++) {  /// loop over candidates
	if (getIsoEmCands) {
	  setInputEmCand((i*4)+(j*2)+k, theLeafCards[i]->getOutputIsoEmCands(j)[k]);
	}
	else {
	  setInputEmCand((i*4)+(j*2)+k, theLeafCards[i]->getOutputNonIsoEmCands(j)[k]);
	}
      }
    }   
  }

}

void L1GctElectronFinalSort::process(){
//Make temporary copy of data
    vector<L1GctEmCand> data = inputCands;
    
//Then sort it
    sort(data.begin(),data.end(),rank_gt());
  
//Copy data to output buffer
    for(int i = 0; i<4; i++){
      outputCands[i] = data[i];
    }
}

void L1GctElectronFinalSort::setInputLeafCard(int i, L1GctEmLeafCard* card) {
  if (i<2) {
    theLeafCards[i] = card;
  }
}

void L1GctElectronFinalSort::setInputEmCand(int i, L1GctEmCand cand){
  inputCands[i] = cand;
}

