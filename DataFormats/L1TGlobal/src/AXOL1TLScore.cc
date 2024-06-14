/**
* \class AXOL1TLScore
*
*
*
* \author: Melissa Quinnan -- UC San Diego
*
*
*/

// this class header
#include "DataFormats/L1TGlobal/interface/AXOL1TLScore.h"

void AXOL1TLScore::reset() {
  axoscore_ = 0.0;
  m_bxInEvent = 0;
}

AXOL1TLScore::AXOL1TLScore(){
  reset();
}

AXOL1TLScore::AXOL1TLScore(int bxNr, int bxInEvent)
  : m_bxInEvent(bxInEvent) {
  axoscore_ = 0.0;
}

//destructor
AXOL1TLScore::~AXOL1TLScore(){
  //empty
}
