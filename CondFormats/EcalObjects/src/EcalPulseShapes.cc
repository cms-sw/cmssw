#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"

EcalPulseShape::EcalPulseShape() {
  for (float& s : pdfval)
    s = 0.;
}
