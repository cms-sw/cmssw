
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
#include<iostream>

using std::cout;

//Overloading the less than operator to use EmCand's
bool compareInputs(L1GctEmCand a, L1GctEmCand b){
  return a.getRank() > b.getRank();
}


L1GctElectronFinalSort::L1GctElectronFinalSort(bool iso):inputCands(0),outputCands(4){
  getIsoEmCands = iso;
}

L1GctElectronFinalSort::~L1GctElectronFinalSort(){
}

void L1GctElectronFinalSort::reset(){
  inputCands.clear();
  outputCands.clear();
}

void L1GctElectronFinalSort::fetchInput(){
  for(vector<L1GctEmLeafCard*>::iterator itLeafCard = theLeafCards.begin();
                                         itLeafCard!=theLeafCards.end();itLeafCard++){ 
    for(unsigned int i=0;i!=theLeafCards.size();i++){
      if(getIsoEmCands){
	vector<L1GctEmCand> isoCands = (*itLeafCard)->getOutputIsoEmCands(i);
	for(unsigned int n=0;n!=isoCands.size();n++){
	  inputCands[n] = isoCands[n];
	}
      }else{
	vector<L1GctEmCand> nonIsoCands = (*itLeafCard)->getOutputNonIsoEmCands(i);
     	for(unsigned int n=0;n!=nonIsoCands.size();n++){
	  inputCands[n] =nonIsoCands[n];
	}   
      }
    }
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
  inputCands[i] = cand;
}

void L1GctElectronFinalSort::setInputLeafCard(int i, L1GctEmLeafCard* card){
  theLeafCards[i] = card;
}

