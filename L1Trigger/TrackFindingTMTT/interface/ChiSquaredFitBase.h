#ifndef L1Trigger_TrackFindingTMTT_ChiSquaredFitBase_h
#define L1Trigger_TrackFindingTMTT_ChiSquaredFitBase_h

///=== This is the base class for the linearised chi-squared track fit algorithms.

///=== Written by: Sioni Summers and Alexander D. Morton.

#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackFitGeneric.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include <TMatrixD.h>
#include <TVectorD.h>

#include <vector>
#include <map>
#include <utility>

namespace tmtt {

  class ChiSquaredFitBase : public TrackFitGeneric {
  public:
    enum PAR_IDS { INVR, PHI0, T, Z0, D0 };

  public:
    ChiSquaredFitBase(const Settings* settings, const uint nPar);

    L1fittedTrack fit(const L1track3D& l1track3D) override;

  protected:
    /* Methods */
    virtual TVectorD seed(const L1track3D& l1track3D) = 0;
    virtual TVectorD residuals(const TVectorD& x) = 0;  // Stub residuals/uncertainty
    virtual TMatrixD D(const TVectorD& x) = 0;          // derivatives
    virtual TMatrixD Vinv() = 0;                        // Covariances

    /* Variables */
    double qOverPt_seed_;
    std::vector<Stub*> stubs_;
    TVectorD trackParams_;
    uint nPar_;
    float largestresid_;
    int ilargestresid_;
    double chiSq_;

  private:
    void calculateChiSq(const TVectorD& resids);
    void calculateDeltaChiSq(const TVectorD& deltaX, const TVectorD& covX);

    int numFittingIterations_;
    int killTrackFitWorstHit_;
    double generalResidualCut_;
    double killingResidualCut_;

    unsigned int minStubLayers_;
    unsigned int minStubLayersRed_;
  };

}  // namespace tmtt

#endif
