/*! \file L1GctElectronFinalSort.cc
 * \Class that does the final sorting of electron candidates
 *
 * This class sorts the electron candidates by rank in 
 * ascending order. Inputs are the 4 highest Et electrons from
 * the leaf? cards
 *
 * \author Maria Hansen
 * \date April 2006
 */

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronFinalSort.h"

L1GctElectronFinalSort::L1GctElectronFinalSort():inputCands(0),outputCands(4)
{
}

L1GctElectronFinalSort::~L1GctElectronFinalSort()
{
}

void L1GctElectronFinalSort::reset() {
	
}

void L1GctElectronFinalSort::fetchInput() {
	
}

void L1GctElectronFinalSort::process() {
	
}

void L1GctElectronFinalSort::setSortedInput(L1GctEmCand cand)
{
  sortedCands.setInputEmCand(cand);
  inputCands.push_back(cand);
}

void L1GctElectronFinalSort::process()
{
  sortedCands.process();
  outputCands = sortedCands.getOutput();
}

void L1GctElectronFinalSort::reset()
{
  //sortedCands.
  inputCands.clear();
  outputCands.clear();
}

