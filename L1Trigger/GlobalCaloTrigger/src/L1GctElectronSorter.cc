/*! \file L1GctElectronSorter.cc
 * \Class that sort electron candidates
 *
 * This class sorts the electron candidates by rank in 
 * ascending order.
 *
 * \author Maria Hansen
 * \date March 2006
 */

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"
#include <iostream>
#include <sort>

using std::cout;
using std::sort;

//Overloading the less than operator to use EmCand's
bool compare(L1GctEmCand a, L1GctEmCand b){
    return a.getRank() > b.getRank();
}

L1GctElectronSorter::L1GctElectronSorter(bool iso):
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
}

// get the input data
void L1GctElectronSorter::fetchInput() {

  // loop over Source Cards
  for (unsigned i=0; i<theSCs.size(); i++) {
    
    // loop over 4 candidates per Source Card
    for (unsigned j=0; j<4; j++) {

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
    sort(data.begin(),data.end(),compare);
  
//Copy data to output buffer
    for(int i = 0; i<4; i++){
      outputCands[i] = data[i];
    }
}

void L1GctElectronSorter::setInputSourceCard(int i, L1GctSourceCard* sc) {
  if (i < theSCs.size()) {
    theSCs[i]=sc;
  }
}

void L1GctElectronSorter::setInputEmCand(L1GctEmCand cand){
  inputCands.push_back(cand);
}

