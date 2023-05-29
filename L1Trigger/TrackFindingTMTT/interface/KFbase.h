#ifndef L1Trigger_TrackFindingTMTT_KFbase_h
#define L1Trigger_TrackFindingTMTT_KFbase_h

#include "L1Trigger/TrackFindingTMTT/interface/TrackFitGeneric.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/KalmanState.h"
#include <TMatrixD.h>
#include <TVectorD.h>

#include <map>
#include <vector>
#include <fstream>
#include <memory>
#include <TString.h>

class TH1F;
class TH2F;

///=== This is the base class for the Kalman Combinatorial Filter track fit algorithm.
///=== All variable names & equations come from Fruhwirth KF paper
///=== http://dx.doi.org/10.1016/0168-9002%2887%2990887-4

namespace tmtt {

  class TP;
  class KalmanState;

  class KFbase : public TrackFitGeneric {
  public:
    enum PAR_IDS { INV2R, PHI0, T, Z0, D0 };
    enum PAR_IDS_VAR { QOVERPT };
    enum MEAS_IDS { PHI, Z };

  public:
    // Initialize configuration
    KFbase(const Settings *settings, const uint nHelixPar, const std::string &fitterName = "", const uint nMeas = 2);

    ~KFbase() override { this->resetStates(); }

    KFbase(const KFbase &kf) = delete;  // Disable copy & move of this class.
    KFbase(KFbase &&kf) = delete;
    KFbase &operator=(const KFbase &kf) = delete;
    KFbase &operator=(KFbase &&kf) = delete;

    // Do KF fit
    L1fittedTrack fit(const L1track3D &l1track3D) override;

    static const unsigned int nKFlayer_ = 7;
    static const unsigned int nEta_ = 16;
    static const unsigned int nGPlayer_ = 7;
    static constexpr unsigned int invalidKFlayer_ = nKFlayer_;

    // index across is GP encoded layer ID (where barrel layers=1,2,7,5,4,3 & endcap wheels=3,4,5,6,7 & 0 never occurs)
    // index down is eta reg
    // element.first is kalman layer when stub is from barrel, 7 is invalid
    // element.second is kalman layer when stub is from endcap, 7 is invalid

    static constexpr std::pair<unsigned, unsigned> layerMap_[nEta_ / 2][nGPlayer_ + 1] = {
        {{7, 7}, {0, 7}, {1, 7}, {5, 7}, {4, 7}, {3, 7}, {7, 7}, {2, 7}},  // B1 B2 B3 B4 B5 B6
        {{7, 7}, {0, 7}, {1, 7}, {5, 7}, {4, 7}, {3, 7}, {7, 7}, {2, 7}},  // B1 B2 B3 B4 B5 B6
        {{7, 7}, {0, 7}, {1, 7}, {5, 7}, {4, 7}, {3, 7}, {7, 7}, {2, 7}},  // B1 B2 B3 B4 B5 B6
        {{7, 7}, {0, 7}, {1, 7}, {5, 7}, {4, 7}, {3, 7}, {7, 7}, {2, 7}},  // B1 B2 B3 B4 B5 B6
        {{7, 7}, {0, 7}, {1, 7}, {5, 4}, {4, 5}, {3, 6}, {7, 7}, {2, 7}},  // B1 B2 B3 B4(/D3) B5(/D2) B6(/D1)
        {{7, 7}, {0, 7}, {1, 7}, {7, 3}, {7, 4}, {2, 5}, {7, 6}, {2, 6}},  // B1 B2 B3(/D5)+B4(/D3) D1 D2 X D4
        {{7, 7}, {0, 7}, {1, 7}, {7, 2}, {7, 3}, {7, 4}, {7, 5}, {7, 6}},  // B1 B2 D1 D2 D3 D4 D5
        {{7, 7}, {0, 7}, {7, 7}, {7, 1}, {7, 2}, {7, 3}, {7, 4}, {7, 5}},  // B1 D1 D2 D3 D4 D5
    };

  protected:
    // Do KF fit (internal)
    const KalmanState *doKF(const L1track3D &l1track3D, const std::vector<Stub *> &stubs, const TP *tpa);

    // Add one stub to a helix state
    virtual const KalmanState *kalmanUpdate(
        unsigned nSkipped, unsigned layer, Stub *stub, const KalmanState *state, const TP *tp);

    // Create a KalmanState, containing a helix state & next stub it is to be updated with.
    const KalmanState *mkState(const L1track3D &candidate,
                               unsigned nSkipped,
                               unsigned layer,
                               const KalmanState *last_state,
                               const TVectorD &x,
                               const TMatrixD &pxx,
                               const TMatrixD &K,
                               const TMatrixD &dcov,
                               Stub *stub,
                               double chi2rphi,
                               double chi2rz);

    //--- Input data

    // Seed track helix params & covariance matrix
    virtual TVectorD seedX(const L1track3D &l1track3D) const = 0;
    virtual TMatrixD seedC(const L1track3D &l1track3D) const = 0;

    // Stub coordinate measurements & resolution
    virtual TVectorD vectorM(const Stub *stub) const = 0;
    virtual TMatrixD matrixV(const Stub *stub, const KalmanState *state) const = 0;

    //--- KF maths matrix multiplications

    // Derivate of helix intercept point w.r.t. helix params.
    virtual TMatrixD matrixH(const Stub *stub) const = 0;
    // Kalman helix ref point extrapolation matrix
    virtual TMatrixD matrixF(const Stub *stub = nullptr, const KalmanState *state = nullptr) const = 0;
    // Product of H*C*H(transpose) (where C = helix covariance matrix)
    TMatrixD matrixHCHt(const TMatrixD &h, const TMatrixD &c) const;
    // Get inverted Kalman R matrix: inverse(V + HCHt)
    TMatrixD matrixRinv(const TMatrixD &matH, const TMatrixD &matCref, const TMatrixD &matV) const;
    // Kalman gain matrix
    TMatrixD getKalmanGainMatrix(const TMatrixD &h, const TMatrixD &pxcov, const TMatrixD &covRinv) const;

    // Residuals of stub with respect to helix.
    virtual TVectorD residual(const Stub *stub, const TVectorD &x, double candQoverPt) const;

    // Update helix state & its covariance matrix with new stub
    void adjustState(const TMatrixD &K,
                     const TMatrixD &pxcov,
                     const TVectorD &x,
                     const TMatrixD &h,
                     const TVectorD &delta,
                     TVectorD &new_x,
                     TMatrixD &new_xcov) const;
    // Update track fit chi2 with new stub
    virtual void adjustChi2(const KalmanState *state,
                            const TMatrixD &covRinv,
                            const TVectorD &delta,
                            double &chi2rphi,
                            double &chi2rz) const;

    //--- Utilities

    // Reset internal data ready for next track.
    void resetStates();

    // Convert to physical helix params instead of local ones used by KF
    virtual TVectorD trackParams(const KalmanState *state) const = 0;
    // Ditto after applying beam-spot constraint.
    virtual TVectorD trackParams_BeamConstr(const KalmanState *state, double &chi2rphi_bcon) const = 0;

    // Get phi of centre of sector containing track.
    double sectorPhi() const {
      float phiCentreSec0 = -M_PI / float(settings_->numPhiNonants()) + M_PI / float(settings_->numPhiSectors());
      return 2. * M_PI * float(iPhiSec_) / float(settings_->numPhiSectors()) + phiCentreSec0;
    }

    // Get KF layer (which is integer representing order layers cross)
    virtual unsigned int kalmanLayer(
        unsigned int iEtaReg, unsigned int layerIDreduced, bool barrel, float r, float z) const;
    // Check if it is unclear whether a particle is expect to cross this layer.
    virtual bool kalmanAmbiguousLayer(unsigned int iEtaReg, unsigned int kfLayer);
    // KF algo mods to cope with dead tracker layers.
    std::set<unsigned> kalmanDeadLayers(bool &remove2PSCut) const;

    // Function to calculate approximation for tilted barrel modules (aka B) copied from Stub class.
    float approxB(float z, float r) const;

    // Is this HLS code?
    virtual bool isHLS() { return false; };

    // Helix state pases cuts.
    virtual bool isGoodState(const KalmanState &state) const = 0;

    //--- Debug printout
    void printTP(const TP *tp) const;
    void printStubLayers(const std::vector<Stub *> &stubs, unsigned int iEtaReg) const;
    void printStub(const Stub *stub) const;
    void printStubs(const std::vector<Stub *> &stubs) const;

  protected:
    unsigned nHelixPar_;
    unsigned nMeas_;
    unsigned numEtaRegions_;

    unsigned int iPhiSec_;
    unsigned int iEtaReg_;

    unsigned int numUpdateCalls_;

    // All helix states KF produces for current track.
    std::vector<std::unique_ptr<const KalmanState>> listAllStates_;

    const TP *tpa_;
  };

}  // namespace tmtt

#endif
