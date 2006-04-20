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

void L1GctElectronFinalSort::reset()
{
  inputCands.clear();
  outputCands.clear();
}

void L1GctElectronFinalSort::fetchInput() {
	
}

void L1GctElectronFinalSort::process()
{
  //sortedCands.process();
  //outputCands = sortedCands.getOutput();
}

void L1GctElectronFinalSort::setInputEmCand(int i, L1GctEmCand cand)
{
  inputCands[i] = cand;
}

