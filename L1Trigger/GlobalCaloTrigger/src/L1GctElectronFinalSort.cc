/*! \file L1GctElectronFinalSort.cc
 * \Class that does the final sorting of electron candidates
 *
 * This class sorts the electron candidates by rank in 
 * ascending order. Inputs are the 4 highest Et electrons from
 * the leaf? cards
 *
 * \author  Maria Hansen
 * \date    21/04/06
 * \version 1.1
 */

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronFinalSort.h"


//Overloading the less than operator to use EmCand's
bool compareInputs(L1GctEmCand a, L1GctEmCand b){
  return a.getRank() > b.getRank();
}


L1GctElectronFinalSort::L1GctElectronFinalSort():inputCands(0),outputCands(4)
{
}

L1GctElectronFinalSort::L1GctElectronFinalSort(vector<L1GctElectronSorter*> src){
  theEmSorters = src;
}

L1GctElectronFinalSort::~L1GctElectronFinalSort(){
}

void L1GctElectronFinalSort::reset(){
  inputCands.clear();
  outputCands.clear();
}

void L1GctElectronFinalSort::fetchInput(){
  for(vector<L1GctElectronSorter*>::iterator itSorted = theEmSorters.begin();itSorted!=theEmSorters.end();itSorted++){ 
    inputCands = (*itSorted)->getOutput();
  }
}

void L1GctElectronFinalSort::process(){
//Make temporary copy of data
    vector<L1GctEmCand> data = inputCands;
    
//Then sort it
    sort(data.begin(),data.end(),compareInputs);
  
//Copy data to output buffer
    for(int i = 0; i<4; i++){
      outputCands[i] = data[i];
    }
}

void L1GctElectronFinalSort::setInputEmCand(int i, L1GctEmCand cand){
  // inputCands[i] = cand;
  //using push_back for now until know how many to set the inputCands vector to in constructor
  inputCands.push_back(cand);
}







