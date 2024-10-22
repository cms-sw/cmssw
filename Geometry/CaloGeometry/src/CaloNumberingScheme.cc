///////////////////////////////////////////////////////////////////////////////
// File: CaloNumberingScheme.cc
// Description: Base class for numbering scheme of calorimeters
///////////////////////////////////////////////////////////////////////////////
#include "Geometry/CaloGeometry/interface/CaloNumberingScheme.h"

CaloNumberingScheme::CaloNumberingScheme(int iv) : verbosity(iv) {}

void CaloNumberingScheme::setVerbosity(const int iv) { verbosity = iv; }
