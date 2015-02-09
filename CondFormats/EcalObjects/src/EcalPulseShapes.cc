#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"

EcalPulseShape::EcalPulseShape() {
  for(int s=0; s<TEMPLATESAMPLES; ++s) pdfval[s] = 0.;
}
