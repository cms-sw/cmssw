#ifndef CalibMuon_DTCalibration_DTRecHitSegmentResidual_h
#define CalibMuon_DTCalibration_DTRecHitSegmentResidual_h

/*
 *  $Date: 2011/02/22 18:43:20 $
 *  $Revision: 1.1 $
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

