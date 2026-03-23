#ifndef CalibMuon_DTCalibration_DTRecHitSegmentResidual_h
#define CalibMuon_DTCalibration_DTRecHitSegmentResidual_h

/*
 *  \author A. Vilela Pereira
 */

#include "DataFormats/DTRecHit/interface/DTRecSegment4DFwd.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DFwd.h"

class DTGeometry;

class DTRecHitSegmentResidual {
public:
  DTRecHitSegmentResidual() {}
  ~DTRecHitSegmentResidual() {}
  float compute(const DTGeometry*, const DTRecHit1D&, const DTRecSegment4D&);

private:
};

#endif
