/*! \file L1GctElectronSorter.cc
 * \Class that sort electron candidates
 *
 * This class sorts the electron candidates by rank in 
 * ascending order.
 *
 * \author Maria Hansen
 * \date March 2006
 */
//Below paths must be changed when scramming is possible
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include<iostream>

using std::cout;

//Overloading the less than operator to use EmCand's
bool compare(L1GctEmCand a, L1GctEmCand b){
    return a.getRank() > b.getRank();
}

L1GctElectronSorter::L1GctElectronSorter():inputCands(0),outputCands(4)
{

}

L1GctElectronSorter::~L1GctElectronSorter()
{

}

void L1GctElectronSorter::setInputEmCand(L1GctEmCand cand){
  inputCands.push_back(cand);
}

//Process sorts the electron candidates after rank and stores the highest four (in the outputCands vector)
void L1GctElectronSorter::process(){

//Make temporary copy of data
    vector<L1GctEmCand> data = inputCands;
    
//Then sort it
    sort(data.begin(),data.end(),compare);
  
//Copy data to output buffer
    for(int i = 0; i<4; i++){
      outputCands[i] = data[i];
    }
}



