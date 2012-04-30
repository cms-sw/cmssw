#ifndef _TRACKER_ALGEBRAICOBJECTS_H_
#define _TRACKER_ALGEBRAICOBJECTS_H_

#ifdef HEP_SHORT_NAMES
#undef HEP_SHORT_NAMES
#endif

// will percolate in CLHEP...
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/SymMatrix.h"

typedef CLHEP::HepVector      AlgebraicVector;
typedef CLHEP::HepMatrix      AlgebraicMatrix;
typedef CLHEP::HepSymMatrix   AlgebraicSymMatrix;

#include "DataFormats/CLHEP/interface/Migration.h" //INCLUDECHECKER:SKIP

#endif
