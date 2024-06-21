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

AXOL1TLScore::AXOL1TLScore() { reset(); }

AXOL1TLScore::AXOL1TLScore(int bxInEvent) : m_bxInEvent(bxInEvent) { axoscore_ = 0.0; }

AXOL1TLScore::AXOL1TLScore(int bxInEvent, float score) : m_bxInEvent(bxInEvent), axoscore_(score) {}

//destructor
AXOL1TLScore::~AXOL1TLScore() {
  //empty
}
