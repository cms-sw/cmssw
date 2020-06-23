#ifndef L1Trigger_TrackFindingTMTT_ChiSquaredFit4_h
#define L1Trigger_TrackFindingTMTT_ChiSquaredFit4_h

#include "L1Trigger/TrackFindingTMTT/interface/ChiSquaredFitBase.h"

namespace tmtt {

  class ChiSquaredFit4 : public ChiSquaredFitBase {
  public:
    ChiSquaredFit4(const Settings* settings, const uint nPar);

  protected:
    TVectorD seed(const L1track3D& l1track3D) override;
    TVectorD residuals(const TVectorD& x) override;
    TMatrixD D(const TVectorD& x) override;
    TMatrixD Vinv() override;
  };

}  // namespace tmtt

#endif
