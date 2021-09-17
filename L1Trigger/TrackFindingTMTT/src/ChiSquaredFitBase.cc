///=== This is the base class for the linearised chi-squared track fit algorithms.

///=== Written by: Sioni Summers and Alexander D. Morton

#include "L1Trigger/TrackFindingTMTT/interface/ChiSquaredFitBase.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"

#include <algorithm>
#include <functional>
#include <set>

namespace tmtt {

  ChiSquaredFitBase::ChiSquaredFitBase(const Settings* settings, const uint nPar)
      : TrackFitGeneric(settings), chiSq_(0.0) {
    // Bad stub killing settings
    numFittingIterations_ = settings_->numTrackFitIterations();
    killTrackFitWorstHit_ = settings_->killTrackFitWorstHit();
    generalResidualCut_ = settings_->generalResidualCut();  // The cut used to remove bad stubs (if nStubs > minLayers)
    killingResidualCut_ = settings_->killingResidualCut();  // The cut used to kill off tracks entirely

    //--- These two parameters are used to check if after the fit, there are still enough stubs on the track
    minStubLayers_ = settings_->minStubLayers();
    nPar_ = nPar;
  }

  void ChiSquaredFitBase::calculateChiSq(const TVectorD& resids) {
    chiSq_ = 0.0;
    uint j = 0;
    for (uint i = 0; i < stubs_.size(); i++) {
      chiSq_ += resids[j] * resids[j] + resids[j + 1] * resids[j + 1];
      j = j + 2;
    }
  }

  void ChiSquaredFitBase::calculateDeltaChiSq(const TVectorD& delX, const TVectorD& covX) {
    for (int i = 0; i < covX.GetNrows(); i++) {
      chiSq_ -= (delX[i]) * covX[i];
    }
  }

  L1fittedTrack ChiSquaredFitBase::fit(const L1track3D& l1track3D) {
    qOverPt_seed_ = l1track3D.qOverPt();
    stubs_ = l1track3D.stubs();

    // Get cut on number of layers including variation due to dead sectors, pt dependence etc.
    minStubLayersRed_ = Utility::numLayerCut(Utility::AlgoStep::FIT,
                                             settings_,
                                             l1track3D.iPhiSec(),
                                             l1track3D.iEtaReg(),
                                             std::abs(l1track3D.qOverPt()),
                                             l1track3D.eta());

    TVectorD x = seed(l1track3D);

    for (int i = 0; i < numFittingIterations_; i++) {
      TMatrixD d = D(x);
      TMatrixD dTrans(TMatrixD::kTransposed, d);
      TMatrixD dtVinv = dTrans * Vinv();
      TMatrixD dtVinvTrans(TMatrixD::kTransposed, dtVinv);
      //TMatrixD M = dtVinv * d; // Must insert extra factor Vinv, due to unconventional Vinv() definition.
      TMatrixD M = dtVinv * dtVinvTrans;
      TMatrixD Minv(TMatrixD::kInverted, M);
      TVectorD resids = residuals(x);
      TVectorD deltaX = Minv * dtVinv * resids;
      x = x - deltaX;
      TVectorD covX = dTrans * Vinv() * resids;
      calculateChiSq(resids);
      calculateDeltaChiSq(deltaX, covX);

      if (i < numFittingIterations_ - 1) {  // Don't kill stub if will not refit.

        resids = residuals(x);  // update resids & largestresid_

        bool killWorstStub = false;
        if (killTrackFitWorstHit_) {
          if (largestresid_ > killingResidualCut_) {
            killWorstStub = true;
          } else if (largestresid_ > generalResidualCut_) {
            std::vector<Stub*> stubsTmp = stubs_;
            stubsTmp.erase(stubsTmp.begin() + ilargestresid_);
            if (Utility::countLayers(settings_, stubsTmp) >= minStubLayersRed_)
              killWorstStub = true;
          } else {
            // Get better apparent tracking performance by always killing worst stub until only 4 layers left.
            if (Utility::countLayers(settings_, stubs_) > minStubLayersRed_)
              killWorstStub = true;
          }
        }

        if (killWorstStub) {
          stubs_.erase(stubs_.begin() + ilargestresid_);

          // Reject tracks with too many killed stubs & stop iterating.
          unsigned int nLayers = Utility::countLayers(settings_, stubs_);  // Count tracker layers with stubs
          bool valid = nLayers >= minStubLayersRed_;

          if (not valid) {
            L1fittedTrack rejectedTrk;
            return rejectedTrk;
          }
        } else {
          break;
        }
      }
    }

    // Reject tracks with too many killed stubs
    unsigned int nLayers = Utility::countLayers(settings_, stubs_);  // Count tracker layers with stubs
    bool valid = nLayers >= minStubLayersRed_;

    if (valid) {
      const unsigned int hitPattern = 0;  // FIX: Needs setting
      const float chi2rz = 0;             // FIX: Needs setting
      return L1fittedTrack(settings_,
                           &l1track3D,
                           stubs_,
                           hitPattern,
                           x[INVR] / (settings_->invPtToInvR()),
                           0,
                           x[PHI0],
                           x[Z0],
                           x[T],
                           chiSq_,
                           chi2rz,
                           nPar_);
    } else {
      L1fittedTrack rejectedTrk;
      return rejectedTrk;
    }
  }

}  // namespace tmtt
