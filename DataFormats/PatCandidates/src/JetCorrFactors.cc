//
// $Id$
//

#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"


using namespace pat;


JetCorrFactors::JetCorrFactors() :
  scaleDefault_(1), scaleUds_(1), scaleGlu_(1), scaleC_(1), scaleB_(1) {
}


JetCorrFactors::JetCorrFactors(float scaleDefault, float scaleUds, float scaleGlu, float scaleC, float scaleB) :
  scaleDefault_(scaleDefault), scaleUds_(scaleUds), scaleGlu_(scaleGlu), scaleC_(scaleC), scaleB_(scaleB) {
}


float JetCorrFactors::scaleDefault() const {
  return scaleDefault_;
}

float JetCorrFactors::scaleUds() const {
  return scaleUds_;
}

float JetCorrFactors::scaleGlu() const {
  return scaleGlu_;
}

float JetCorrFactors::scaleC() const {
  return scaleC_;
}

float JetCorrFactors::scaleB() const {
  return scaleB_;
}

