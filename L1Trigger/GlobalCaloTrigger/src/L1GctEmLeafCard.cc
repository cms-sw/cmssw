#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmLeafCard.h"

#include <vector>

L1GctEmLeafCard::L1GctEmLeafCard() :
  theEmSorters(4) 
{
}

L1GctEmLeafCard::~L1GctEmLeafCard() {
}


/// clear buffers
void L1GctEmLeafCard::reset() {
  for (unsigned i=0; i<theEmSorters.size(); i++) {
    theEmSorters[i]->reset();
  }
}

/// fetch input data
void L1GctEmLeafCard::fetchInput() {
  for (unsigned i=0; i<theEmSorters.size(); i++) {
    theEmSorters[i]->fetchInput();
  }
}

/// process the event
void L1GctEmLeafCard::process() {
  for (unsigned i=0; i<theEmSorters.size(); i++) {
    theEmSorters[i]->process();
  }
}

/// add a source card as input
void L1GctEmLeafCard::addInputSourceCard(L1GctSourceCard* card) {

}

/// get the output candidates
vector<L1GctEmCand> L1GctEmLeafCard::getOutputIsoEmCands() {
   return theEmSorters[0]->getOutput();
}

/// get the output candidates
vector<L1GctEmCand> L1GctEmLeafCard::getOutputNonIsoEmCands() {
     return theEmSorters[1]->getOutput();
}
