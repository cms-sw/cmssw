#include "L1Trigger/TrackFindingTMTT/interface/KalmanState.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "TMatrixD.h"

using namespace std;

namespace tmtt {

  KalmanState::KalmanState(const Settings *settings,
                           const L1track3D &candidate,
                           unsigned nSkipped,
                           int kLayer,
                           const KalmanState *last_state,
                           const TVectorD &vecX,
                           const TMatrixD &matC,
                           const TMatrixD &matK,
                           const TMatrixD &matV,
                           Stub *stub,
                           double chi2rphi,
                           double chi2rz)
      : settings_(settings),
        kLayer_(kLayer),
        last_state_(last_state),
        vecX_(vecX),
        stub_(stub),
        chi2rphi_(chi2rphi),
        chi2rz_(chi2rz),
        nSkipped_(nSkipped),
        l1track3D_(candidate) {
    matC_.Clear();
    matC_.ResizeTo(matC.GetNrows(), matC.GetNcols());
    matC_ = matC;
    matK_.ResizeTo(matK.GetNrows(), matK.GetNcols());
    matK_ = matK;
    matV_.ResizeTo(matV.GetNrows(), matV.GetNcols());
    matV_ = matV;
    kalmanChi2RphiScale_ = settings_->kalmanChi2RphiScale();

    hitPattern_ = 0;
    if (last_state != nullptr)
      hitPattern_ = last_state->hitPattern();  // Bit encoded list of hit layers
    if (stub != nullptr && kLayer_ >= 0)
      hitPattern_ |= (1 << (kLayer_));

    r_ = 0.1;
    z_ = 0;
    barrel_ = true;

    if (stub != nullptr) {
      r_ = stub->r();
      z_ = stub->z();
      barrel_ = stub->barrel();
    }

    n_stubs_ = 1 + kLayer_ - nSkipped_;
  }

  bool KalmanState::good(const TP *tp) const {
    const KalmanState *state = this;
    while (state) {
      Stub *stub = state->stub();
      if (stub != nullptr) {
        const set<const TP *> &tps = stub->assocTPs();
        if (tps.find(tp) == tps.end())
          return false;
      }
      state = state->last_state();
    }
    return true;
  }

  double KalmanState::reducedChi2() const {
    if (2 * n_stubs_ - vecX_.GetNrows() > 0)
      return (this->chi2()) / (2 * n_stubs_ - vecX_.GetNrows());
    else
      return 0;
  }

  const KalmanState *KalmanState::last_update_state() const {
    const KalmanState *state = this;
    while (state) {
      if (state->stub() != nullptr)
        return state;
      state = state->last_state();
    }
    return nullptr;
  }

  std::vector<Stub *> KalmanState::stubs() const {
    std::vector<Stub *> all_stubs;

    const KalmanState *state = this;
    while (state) {
      Stub *stub = state->stub();
      if (stub != nullptr)
        all_stubs.push_back(stub);
      state = state->last_state();
    }
    std::reverse(all_stubs.begin(), all_stubs.end());  // Put innermost stub first.
    return all_stubs;
  }

}  // namespace tmtt
