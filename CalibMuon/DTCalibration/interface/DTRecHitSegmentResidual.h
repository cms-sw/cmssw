#ifndef CalibMuon_DTCalibration_DTRecHitSegmentResidual_h
#define CalibMuon_DTCalibration_DTRecHitSegmentResidual_h

/*
 *  \author A. Vilela Pereira
 */

#include "DataFormats/DTRecHit/interface/DTRecSegment4DFwd.h"

class DTGeometry;
class DTRecHit1D;

class DTRecHitSegmentResidual {
public:
  DTRecHitSegmentResidual() {}
  ~DTRecHitSegmentResidual() {}
  float compute(const DTGeometry*, const DTRecHit1D&, const DTRecSegment4D&);

private:
};

#endif
