#ifndef CalibMuon_DTCalibration_DTRecHitSegmentResidual_h
#define CalibMuon_DTCalibration_DTRecHitSegmentResidual_h

/*
 *  \author A. Vilela Pereira
 */

class DTGeometry;
class DTRecSegment4D;
class DTRecHit1D;

class DTRecHitSegmentResidual {
public:
  DTRecHitSegmentResidual() {}
  ~DTRecHitSegmentResidual() {}
  float compute(const DTGeometry*, const DTRecHit1D&, const DTRecSegment4D&);

private:
};

#endif
