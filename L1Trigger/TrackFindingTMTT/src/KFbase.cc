///=== This is the base class for the Kalman Combinatorial Filter track fit algorithm.

///=== Written by: S. Summers, K. Uchida, M. Pesaresi, I.Tomalin

#include "L1Trigger/TrackFindingTMTT/interface/KFbase.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/KalmanState.h"
#include "L1Trigger/TrackFindingTMTT/interface/StubKiller.h"
#include "L1Trigger/TrackFindingTMTT/interface/PrintL1trk.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "TMatrixD.h"

#include <algorithm>
#include <functional>
#include <fstream>
#include <iomanip>
#include <atomic>
#include <sstream>

using namespace std;

namespace tmtt {

  /* Initialize cfg parameters */

  KFbase::KFbase(const Settings *settings, const uint nHelixPar, const string &fitterName, const uint nMeas)
      : TrackFitGeneric(settings, fitterName) {
    nHelixPar_ = nHelixPar;
    nMeas_ = nMeas;
    numEtaRegions_ = settings->numEtaRegions();
  }

  /* Do track fit */

  L1fittedTrack KFbase::fit(const L1track3D &l1track3D) {
    iPhiSec_ = l1track3D.iPhiSec();
    iEtaReg_ = l1track3D.iEtaReg();
    resetStates();
    numUpdateCalls_ = 0;

    vector<Stub *> stubs = l1track3D.stubs();

    auto orderByLayer = [](const Stub *a, const Stub *b) { return bool(a->layerId() < b->layerId()); };
    sort(stubs.begin(), stubs.end(), orderByLayer);  // Makes debug printout pretty.

    //TP
    const TP *tpa(nullptr);
    if (l1track3D.matchedTP()) {
      tpa = l1track3D.matchedTP();
    }
    tpa_ = tpa;

    //track information dump
    if (settings_->kalmanDebugLevel() >= 1) {
      PrintL1trk() << "===============================================================================";
      std::stringstream text;
      text << std::fixed << std::setprecision(4);
      text << "Input track cand: [phiSec,etaReg]=[" << l1track3D.iPhiSec() << "," << l1track3D.iEtaReg() << "]";
      text << " HT(m,c)=(" << l1track3D.cellLocationHT().first << "," << l1track3D.cellLocationHT().second
           << ") q/pt=" << l1track3D.qOverPt() << " tanL=" << l1track3D.tanLambda() << " z0=" << l1track3D.z0()
           << " phi0=" << l1track3D.phi0() << " nStubs=" << l1track3D.numStubs() << " d0=" << l1track3D.d0();
      PrintL1trk() << text.str();
      if (not settings_->hybrid())
        printTP(tpa);
      if (settings_->kalmanDebugLevel() >= 2) {
        printStubLayers(stubs, l1track3D.iEtaReg());
        printStubs(stubs);
      }
    }

    //Kalman Filter
    const KalmanState *cand = doKF(l1track3D, stubs, tpa);

    //return L1fittedTrk for the selected state (if KF produced one it was happy with).
    if (cand != nullptr) {
      // Get track helix params.
      TVectorD trackPars = trackParams(cand);
      double d0 = (nHelixPar_ == 5) ? trackPars[D0] : 0.;

      L1fittedTrack fitTrk(settings_,
                           &l1track3D,
                           cand->stubs(),
                           cand->hitPattern(),
                           trackPars[QOVERPT],
                           d0,
                           trackPars[PHI0],
                           trackPars[Z0],
                           trackPars[T],
                           cand->chi2rphi(),
                           cand->chi2rz(),
                           nHelixPar_);

      // Store supplementary info, specific to KF fitter.
      fitTrk.setInfoKF(cand->nSkippedLayers(), numUpdateCalls_);

      // If doing 5 parameter fit, optionally also calculate helix params & chi2 with beam-spot constraint applied,
      // and store inside L1fittedTrack object.
      if (settings_->kalmanAddBeamConstr()) {
        if (nHelixPar_ == 5) {
          double chi2rphi_bcon = 0.;
          TVectorD trackPars_bcon = trackParams_BeamConstr(cand, chi2rphi_bcon);

          // Check scaled chi2 cut
          vector<double> kfLayerVsChiSqCut = settings_->kfLayerVsChiSq5();
          double chi2scaled = chi2rphi_bcon / settings_->kalmanChi2RphiScale() + fitTrk.chi2rz();
          bool accepted = true;
          if (chi2scaled > kfLayerVsChiSqCut[cand->nStubLayers()])
            accepted = false;

          fitTrk.setBeamConstr(trackPars_bcon[QOVERPT], trackPars_bcon[PHI0], chi2rphi_bcon, accepted);
        }
      }

      // Fitted track params must lie in same sector as HT originally found track in.
      if (!settings_->hybrid()) {  // consistentSector() function not yet working for Hybrid.

        // Bodge to take into account digitisation in sector consistency check.
        if (settings_->enableDigitize())
          fitTrk.digitizeTrack("KF4ParamsComb");

        if (!fitTrk.consistentSector()) {
          if (settings_->kalmanDebugLevel() >= 1)
            PrintL1trk() << "Track rejected by sector consistency test";
          L1fittedTrack rejectedTrk;
          return rejectedTrk;
        }
      }

      return fitTrk;

    } else {  // Track rejected by fitter

      if (settings_->kalmanDebugLevel() >= 1) {
        bool goodTrack = (tpa && tpa->useForAlgEff());  // Matches truth particle.
        if (goodTrack) {
          int tpin = tpa->index();
          PrintL1trk() << "TRACK LOST: eta=" << l1track3D.iEtaReg() << " pt=" << l1track3D.pt() << " tp=" << tpin;

          for (auto stub : stubs) {
            int kalmanLay =
                this->kalmanLayer(l1track3D.iEtaReg(), stub->layerIdReduced(), stub->barrel(), stub->r(), stub->z());
            std::stringstream text;
            text << std::fixed << std::setprecision(4);
            text << "    Stub: lay_red=" << stub->layerIdReduced() << " KFlay=" << kalmanLay << " r=" << stub->r()
                 << " z=" << stub->z() << "   assoc TPs =";
            for (const TP *tp_i : stub->assocTPs())
              text << " " << tp_i->index();
            PrintL1trk() << text.str();
            if (stub->assocTPs().empty())
              PrintL1trk() << " none";
          }
          PrintL1trk() << "=====================";
        }
      }

      //dump on the missed TP for efficiency calculation.
      if (settings_->kalmanDebugLevel() >= 3) {
        if (tpa && tpa->useForAlgEff()) {
          PrintL1trk() << "TP for eff. missed addr. index : " << tpa << " " << tpa->index();
          printStubs(stubs);
        }
      }

      L1fittedTrack rejectedTrk;
      return rejectedTrk;
    }
  }

  /* Do track fit (internal function) */

  const KalmanState *KFbase::doKF(const L1track3D &l1track3D, const vector<Stub *> &stubs, const TP *tpa) {
    const KalmanState *finished_state = nullptr;

    map<unsigned int, const KalmanState *, std::greater<unsigned int>>
        best_state_by_nstubs;  // Best state (if any) for each viable no. of stubs on track value.

    // seed helix params & their covariance.
    TVectorD x0 = seedX(l1track3D);
    TMatrixD pxx0 = seedC(l1track3D);
    TMatrixD K(nHelixPar_, 2);
    TMatrixD dcov(2, 2);

    const KalmanState *state0 = mkState(l1track3D, 0, -1, nullptr, x0, pxx0, K, dcov, nullptr, 0, 0);

    // internal containers - i.e. the state FIFO. Contains estimate of helix params in last/next layer, with multiple entries if there were multiple stubs, yielding multiple states.
    vector<const KalmanState *> new_states;
    vector<const KalmanState *> prev_states;
    prev_states.push_back(state0);

    // Get dead layers, if any.
    bool remove2PSCut = settings_->kalmanRemove2PScut();
    set<unsigned> kfDeadLayers = kalmanDeadLayers(remove2PSCut);

    // arrange stubs into Kalman layers according to eta region
    int etaReg = l1track3D.iEtaReg();
    map<int, vector<Stub *>> layerStubs;

    for (auto stub : stubs) {
      // Get Kalman encoded layer ID for this stub.
      int kalmanLay = this->kalmanLayer(etaReg, stub->layerIdReduced(), stub->barrel(), stub->r(), stub->z());

      constexpr unsigned int invalidKFlayer = 7;
      if (kalmanLay != invalidKFlayer) {
        if (layerStubs[kalmanLay].size() < settings_->kalmanMaxStubsPerLayer()) {
          layerStubs[kalmanLay].push_back(stub);
        } else {
          // If too many stubs, FW keeps the last stub.
          layerStubs[kalmanLay].back() = stub;
        }
      }
    }

    // iterate using state->nextLayer() to determine next Kalman layer(s) to add stubs from
    constexpr unsigned int nTypicalLayers = 6;  // Number of tracker layers a typical track can pass through.
    // If user asked to add up to 7 layers to track, increase number of iterations by 1.
    const unsigned int maxIterations = std::max(nTypicalLayers, settings_->kalmanMaxNumStubs());
    for (unsigned iteration = 0; iteration < maxIterations; iteration++) {
      int combinations_per_iteration = 0;

      bool easy = (l1track3D.numStubs() < settings_->kalmanMaxStubsEasy());
      unsigned int kalmanMaxSkipLayers =
          easy ? settings_->kalmanMaxSkipLayersEasy() : settings_->kalmanMaxSkipLayersHard();

      // update each state from previous iteration (or seed) using stubs in next Kalman layer
      vector<const KalmanState *>::const_iterator i_state = prev_states.begin();
      for (; i_state != prev_states.end(); i_state++) {
        const KalmanState *the_state = *i_state;

        unsigned int layer = the_state->nextLayer();  // Get KF layer where stubs to be searched for next
        unsigned nSkipped = the_state->nSkippedLayers();

        // If this layer is known to be dead, skip to the next layer (layer+1)
        // The next_states_skipped will then look at layer+2
        // However, if there are stubs in this layer, then don't skip (e.g. our phi/eta boundaries might not line up exactly with a dead region)
        // Continue to skip until you reach a functioning layer (or a layer with stubs)
        unsigned nSkippedDeadLayers = 0;
        unsigned nSkippedAmbiguousLayers = 0;
        while (kfDeadLayers.find(layer) != kfDeadLayers.end() && layerStubs[layer].empty()) {
          layer += 1;
          ++nSkippedDeadLayers;
        }
        while (this->kalmanAmbiguousLayer(etaReg, layer) && layerStubs[layer].empty()) {
          layer += 1;
          ++nSkippedAmbiguousLayers;
        }

        // containers for updated state+stub combinations
        vector<const KalmanState *> next_states;
        vector<const KalmanState *> next_states_skipped;

        // find stubs for this layer
        // (If layer > 6, this will return empty vector, so safe).
        vector<Stub *> thislay_stubs = layerStubs[layer];

        // find stubs for next layer if we skip a layer, except when we are on the penultimate layer,
        // or we have exceeded the max skipped layers
        vector<Stub *> nextlay_stubs;

        // If the next layer (layer+1) is a dead layer, then proceed to the layer after next (layer+2), if possible
        // Also note if we need to increase "skipped" by one more for these states
        unsigned nSkippedDeadLayers_nextStubs = 0;
        unsigned nSkippedAmbiguousLayers_nextStubs = 0;
        if (nSkipped < kalmanMaxSkipLayers) {
          if (kfDeadLayers.find(layer + 1) != kfDeadLayers.end() && layerStubs[layer + 1].empty()) {
            nextlay_stubs = layerStubs[layer + 2];
            nSkippedDeadLayers_nextStubs++;
          } else if (this->kalmanAmbiguousLayer(etaReg, layer) && layerStubs[layer + 1].empty()) {
            nextlay_stubs = layerStubs[layer + 2];
            nSkippedAmbiguousLayers_nextStubs++;
          } else {
            nextlay_stubs = layerStubs[layer + 1];
          }
        }

        // If track was not rejected by isGoodState() is previous iteration, failure here usually means the tracker ran out of layers to explore.
        // (Due to "kalmanLay" not having unique ID for each layer within a given eta sector).
        if (settings_->kalmanDebugLevel() >= 2 && best_state_by_nstubs.empty() && thislay_stubs.empty() &&
            nextlay_stubs.empty())
          PrintL1trk() << "State is lost by start of iteration " << iteration
                       << " : #thislay_stubs=" << thislay_stubs.size() << " #nextlay_stubs=" << nextlay_stubs.size()
                       << " layer=" << layer << " eta=" << l1track3D.iEtaReg();

        // If we skipped over a dead layer, only increment "nSkipped" after the stubs in next+1 layer have been obtained
        nSkipped += nSkippedDeadLayers;
        nSkipped += nSkippedAmbiguousLayers;

        // check to guarantee no fewer than 2PS hits per state at iteration 1
        // (iteration 0 will always include a PS hit, but iteration 1 could use 2S hits
        // unless we include this)
        if (iteration == 1 && !remove2PSCut) {
          vector<Stub *> temp_thislaystubs;
          vector<Stub *> temp_nextlaystubs;
          for (auto stub : thislay_stubs) {
            if (stub->psModule())
              temp_thislaystubs.push_back(stub);
          }
          for (auto stub : nextlay_stubs) {
            if (stub->psModule())
              temp_nextlaystubs.push_back(stub);
          }
          thislay_stubs = temp_thislaystubs;
          nextlay_stubs = temp_nextlaystubs;
        }

        combinations_per_iteration += thislay_stubs.size() + nextlay_stubs.size();

        // loop over each stub in this layer and check for compatibility with this state
        for (unsigned i = 0; i < thislay_stubs.size(); i++) {
          Stub *stub = thislay_stubs[i];

          // Update helix params by adding this stub.
          const KalmanState *new_state = kalmanUpdate(nSkipped, layer, stub, the_state, tpa);

          // Cut on track chi2, pt etc.
          if (isGoodState(*new_state))
            next_states.push_back(new_state);
        }

        // loop over each stub in next layer if we skip, and check for compatibility with this state
        for (unsigned i = 0; i < nextlay_stubs.size(); i++) {
          Stub *stub = nextlay_stubs[i];

          const KalmanState *new_state =
              kalmanUpdate(nSkipped + 1 + nSkippedDeadLayers_nextStubs + nSkippedAmbiguousLayers_nextStubs,
                           layer + 1 + nSkippedDeadLayers_nextStubs + nSkippedAmbiguousLayers_nextStubs,
                           stub,
                           the_state,
                           tpa);

          if (isGoodState(*new_state))
            next_states_skipped.push_back(new_state);
        }

        // post Kalman filter local sorting per state
        auto orderByChi2 = [](const KalmanState *a, const KalmanState *b) {
          return bool(a->chi2scaled() < b->chi2scaled());
        };
        sort(next_states.begin(), next_states.end(), orderByChi2);
        sort(next_states_skipped.begin(), next_states_skipped.end(), orderByChi2);

        new_states.insert(new_states.end(), next_states.begin(), next_states.end());
        new_states.insert(new_states.end(), next_states_skipped.begin(), next_states_skipped.end());
        /*
        i = 0;
        for (auto state : next_states) {
            new_states.push_back(state);
          i++;
        }

        i = 0;
        for (auto state : next_states_skipped) {
            new_states.push_back(state);
          i++;
        }
*/
      }  //end of state loop

      // copy new_states into prev_states for next iteration or end if we are on
      // last iteration by clearing all states and making final state selection

      auto orderByMinSkipChi2 = [](const KalmanState *a, const KalmanState *b) {
        return bool((a->chi2scaled()) * (a->nSkippedLayers() + 1) < (b->chi2scaled()) * (b->nSkippedLayers() + 1));
      };
      sort(new_states.begin(), new_states.end(), orderByMinSkipChi2);  // Sort by chi2*(skippedLayers+1)

      unsigned int nStubs = iteration + 1;
      // Success. We have at least one state that passes all cuts. Save best state found with this number of stubs.
      if (nStubs >= settings_->kalmanMinNumStubs() && not new_states.empty())
        best_state_by_nstubs[nStubs] = new_states[0];

      if (nStubs == settings_->kalmanMaxNumStubs()) {
        // We're done.
        prev_states.clear();
        new_states.clear();

      } else {
        // Continue iterating.
        prev_states = new_states;
        new_states.clear();
      }
    }

    if (not best_state_by_nstubs.empty()) {
      // Select state with largest number of stubs.
      finished_state = best_state_by_nstubs.begin()->second;  // First element has largest number of stubs.
      if (settings_->kalmanDebugLevel() >= 1) {
        std::stringstream text;
        text << std::fixed << std::setprecision(4);
        text << "Track found! final state selection: nLay=" << finished_state->nStubLayers()
             << " hitPattern=" << std::hex << finished_state->hitPattern() << std::dec
             << " phiSec=" << l1track3D.iPhiSec() << " etaReg=" << l1track3D.iEtaReg() << " HT(m,c)=("
             << l1track3D.cellLocationHT().first << "," << l1track3D.cellLocationHT().second << ")";
        TVectorD y = trackParams(finished_state);
        text << " q/pt=" << y[QOVERPT] << " tanL=" << y[T] << " z0=" << y[Z0] << " phi0=" << y[PHI0];
        if (nHelixPar_ == 5)
          text << " d0=" << y[D0];
        text << " chosen from states:";
        for (const auto &p : best_state_by_nstubs)
          text << " " << p.second->chi2() << "/" << p.second->nStubLayers();
        PrintL1trk() << text.str();
      }
    } else {
      if (settings_->kalmanDebugLevel() >= 1) {
        PrintL1trk() << "Track lost";
      }
    }

    return finished_state;
  }

  /*--- Update a helix state by adding a stub. */

  const KalmanState *KFbase::kalmanUpdate(
      unsigned nSkipped, unsigned int layer, Stub *stub, const KalmanState *state, const TP *tpa) {
    if (settings_->kalmanDebugLevel() >= 4) {
      PrintL1trk() << "---------------";
      PrintL1trk() << "kalmanUpdate";
      PrintL1trk() << "---------------";
      printStub(stub);
    }

    numUpdateCalls_++;  // For monitoring, count calls to updator per track.

    // Helix params & their covariance.
    TVectorD vecX = state->vectorX();
    TMatrixD matC = state->matrixC();
    if (state->barrel() && !stub->barrel()) {
      if (settings_->kalmanDebugLevel() >= 4) {
        PrintL1trk() << "STATE BARREL TO ENDCAP BEFORE ";
        PrintL1trk() << "state : " << vecX[0] << " " << vecX[1] << " " << vecX[2] << " " << vecX[3];
        PrintL1trk() << "cov(x): ";
        matC.Print();
      }
      if (settings_->kalmanDebugLevel() >= 4) {
        PrintL1trk() << "STATE BARREL TO ENDCAP AFTER ";
        PrintL1trk() << "state : " << vecX[0] << " " << vecX[1] << " " << vecX[2] << " " << vecX[3];
        PrintL1trk() << "cov(x): ";
        matC.Print();
      }
    }
    // Matrix to propagate helix reference point from one layer to next.
    TMatrixD matF = matrixF(stub, state);
    TMatrixD matFtrans(TMatrixD::kTransposed, matF);
    if (settings_->kalmanDebugLevel() >= 4) {
      PrintL1trk() << "matF";
      matF.Print();
    }

    // Multiply matrices to get helix params relative to reference point at next layer.
    TVectorD vecXref = matF * vecX;
    if (settings_->kalmanDebugLevel() >= 4) {
      PrintL1trk() << "vecFref = [";
      for (unsigned i = 0; i < nHelixPar_; i++)
        PrintL1trk() << vecXref[i] << ", ";
      PrintL1trk() << "]";
    }

    // Get stub residuals.
    TVectorD delta = residual(stub, vecXref, state->candidate().qOverPt());
    if (settings_->kalmanDebugLevel() >= 4) {
      PrintL1trk() << "delta = " << delta[0] << ", " << delta[1];
    }

    // Derivative of predicted (phi,z) intercept with layer w.r.t. helix params.
    TMatrixD matH = matrixH(stub);
    if (settings_->kalmanDebugLevel() >= 4) {
      PrintL1trk() << "matH";
      matH.Print();
    }

    if (settings_->kalmanDebugLevel() >= 4) {
      PrintL1trk() << "previous state covariance";
      matC.Print();
    }
    // Get scattering contribution to helix parameter covariance (currently zero).
    TMatrixD matScat(nHelixPar_, nHelixPar_);

    // Get covariance on helix parameters at new reference point including scattering..
    TMatrixD matCref = matF * matC * matFtrans + matScat;
    if (settings_->kalmanDebugLevel() >= 4) {
      PrintL1trk() << "matCref";
      matCref.Print();
    }
    // Get hit position covariance matrix.
    TMatrixD matV = matrixV(stub, state);
    if (settings_->kalmanDebugLevel() >= 4) {
      PrintL1trk() << "matV";
      matV.Print();
    }

    TMatrixD matRinv = matrixRinv(matH, matCref, matV);
    if (settings_->kalmanDebugLevel() >= 4) {
      PrintL1trk() << "matRinv";
      matRinv.Print();
    }

    // Calculate Kalman Gain matrix.
    TMatrixD matK = getKalmanGainMatrix(matH, matCref, matRinv);
    if (settings_->kalmanDebugLevel() >= 4) {
      PrintL1trk() << "matK";
      matK.Print();
    }

    // Update helix state & its covariance matrix with new stub.
    TVectorD new_vecX(nHelixPar_);
    TMatrixD new_matC(nHelixPar_, nHelixPar_);
    adjustState(matK, matCref, vecXref, matH, delta, new_vecX, new_matC);

    // Update track fit chi2 with new stub.
    double new_chi2rphi = 0, new_chi2rz = 0;
    this->adjustChi2(state, matRinv, delta, new_chi2rphi, new_chi2rz);

    if (settings_->kalmanDebugLevel() >= 4) {
      if (nHelixPar_ == 4)
        PrintL1trk() << "adjusted x = " << new_vecX[0] << ", " << new_vecX[1] << ", " << new_vecX[2] << ", "
                     << new_vecX[3];
      else if (nHelixPar_ == 5)
        PrintL1trk() << "adjusted x = " << new_vecX[0] << ", " << new_vecX[1] << ", " << new_vecX[2] << ", "
                     << new_vecX[3] << ", " << new_vecX[4];
      PrintL1trk() << "adjusted C ";
      new_matC.Print();
      PrintL1trk() << "adjust chi2rphi=" << new_chi2rphi << " chi2rz=" << new_chi2rz;
    }

    const KalmanState *new_state = mkState(
        state->candidate(), nSkipped, layer, state, new_vecX, new_matC, matK, matV, stub, new_chi2rphi, new_chi2rz);

    return new_state;
  }

  /* Create a KalmanState, containing a helix state & next stub it is to be updated with. */

  const KalmanState *KFbase::mkState(const L1track3D &candidate,
                                     unsigned nSkipped,
                                     unsigned layer,
                                     const KalmanState *last_state,
                                     const TVectorD &vecX,
                                     const TMatrixD &matC,
                                     const TMatrixD &matK,
                                     const TMatrixD &matV,
                                     Stub *stub,
                                     double chi2rphi,
                                     double chi2rz) {
    auto new_state = std::make_unique<const KalmanState>(
        settings_, candidate, nSkipped, layer, last_state, vecX, matC, matK, matV, stub, chi2rphi, chi2rz);

    const KalmanState *p_new_state = new_state.get();
    listAllStates_.push_back(std::move(new_state));  // Vector keeps ownership of all states.
    return p_new_state;
  }

  /* Product of H*C*H(transpose) (where C = helix covariance matrix) */

  TMatrixD KFbase::matrixHCHt(const TMatrixD &matH, const TMatrixD &matC) const {
    TMatrixD matHtrans(TMatrixD::kTransposed, matH);
    return matH * matC * matHtrans;
  }

  /* Get inverted Kalman R matrix: inverse(V + HCHt) */

  TMatrixD KFbase::matrixRinv(const TMatrixD &matH, const TMatrixD &matCref, const TMatrixD &matV) const {
    TMatrixD matHCHt = matrixHCHt(matH, matCref);
    TMatrixD matR = matV + matHCHt;
    TMatrixD matRinv(2, 2);
    if (matR.Determinant() > 0) {
      matRinv = TMatrixD(TMatrixD::kInverted, matR);
    } else {
      // Protection against rare maths instability.
      const TMatrixD unitMatrix(TMatrixD::kUnit, TMatrixD(nHelixPar_, nHelixPar_));
      const double big = 9.9e9;
      matRinv = big * unitMatrix;
    }
    if (settings_->kalmanDebugLevel() >= 4) {
      PrintL1trk() << "matHCHt";
      matHCHt.Print();
      PrintL1trk() << "matR";
      matR.Print();
    }
    return matRinv;
  }

  /* Determine Kalman gain matrix K */

  TMatrixD KFbase::getKalmanGainMatrix(const TMatrixD &matH, const TMatrixD &matCref, const TMatrixD &matRinv) const {
    TMatrixD matHtrans(TMatrixD::kTransposed, matH);
    TMatrixD matCrefht = matCref * matHtrans;
    TMatrixD matK = matCrefht * matRinv;
    return matK;
  }

  /* Calculate stub residual w.r.t. helix */

  TVectorD KFbase::residual(const Stub *stub, const TVectorD &vecX, double candQoverPt) const {
    TVectorD vd = vectorM(stub);  // Get (phi relative to sector, z) of hit.
    TMatrixD h = matrixH(stub);
    TVectorD hx = h * vecX;  // Get intercept of helix with layer (linear approx).
    TVectorD delta = vd - hx;

    // Calculate higher order corrections to residuals.

    if (not settings_->kalmanHOfw()) {
      TVectorD correction(2);

      float inv2R = (settings_->invPtToInvR()) * 0.5 * candQoverPt;
      float tanL = vecX[T];
      float z0 = vecX[Z0];

      float deltaS = 0.;
      if (settings_->kalmanHOhelixExp()) {
        // Higher order correction correction to circle expansion for improved accuracy at low Pt.
        double corr = stub->r() * inv2R;

        // N.B. In endcap 2S, this correction to correction[0] is exactly cancelled by the deltaS-dependent correction to it below.
        correction[0] += (1. / 6.) * pow(corr, 3);

        deltaS = (1. / 6.) * (stub->r()) * pow(corr, 2);
        correction[1] -= deltaS * tanL;
      }

      if ((not stub->barrel()) && not(stub->psModule())) {
        // These corrections rely on inside --> outside tracking, so r-z track params in 2S modules known.
        float rShift = (stub->z() - z0) / tanL - stub->r();

        if (settings_->kalmanHOhelixExp())
          rShift -= deltaS;

        if (settings_->kalmanHOprojZcorr() == 1) {
          // Add correlation term related to conversion of stub residuals from (r,phi) to (z,phi).
          correction[0] += inv2R * rShift;
        }

        if (settings_->kalmanHOalpha() == 1) {
          // Add alpha correction for non-radial 2S endcap strips..
          correction[0] += stub->alpha() * rShift;
        }
      }

      // Apply correction to residuals.
      delta += correction;
    }

    delta[0] = reco::deltaPhi(delta[0], 0.);

    return delta;
  }

  /* Update helix state & its covariance matrix with new stub */

  void KFbase::adjustState(const TMatrixD &matK,
                           const TMatrixD &matCref,
                           const TVectorD &vecXref,
                           const TMatrixD &matH,
                           const TVectorD &delta,
                           TVectorD &new_vecX,
                           TMatrixD &new_matC) const {
    new_vecX = vecXref + matK * delta;
    const TMatrixD unitMatrix(TMatrixD::kUnit, TMatrixD(nHelixPar_, nHelixPar_));
    TMatrixD tmp = unitMatrix - matK * matH;
    new_matC = tmp * matCref;
  }

  /* Update track fit chi2 with new stub */

  void KFbase::adjustChi2(const KalmanState *state,
                          const TMatrixD &matRinv,
                          const TVectorD &delta,
                          double &chi2rphi,
                          double &chi2rz) const {
    // Change in chi2 (with r-phi/r-z correlation term included in r-phi component)
    double delChi2rphi = delta[PHI] * delta[PHI] * matRinv[PHI][PHI] + 2 * delta[PHI] * delta[Z] * matRinv[PHI][Z];
    double delChi2rz = delta[Z] * delta[Z] * matRinv[Z][Z];

    if (settings_->kalmanDebugLevel() >= 4) {
      PrintL1trk() << "delta(chi2rphi)=" << delChi2rphi << " delta(chi2rz)= " << delChi2rz;
    }
    chi2rphi = state->chi2rphi() + delChi2rphi;
    chi2rz = state->chi2rz() + delChi2rz;
    return;
  }

  /* Reset internal data ready for next track. */

  void KFbase::resetStates() { listAllStates_.clear(); }

  /* Get Kalman layer mapping (i.e. layer order in which stubs should be processed) */

  unsigned int KFbase::kalmanLayer(
      unsigned int iEtaReg, unsigned int layerIDreduced, bool barrel, float r, float z) const {
    // index across is GP encoded layer ID (where barrel layers=1,2,7,5,4,3 & endcap wheels=3,4,5,6,7 & 0 never occurs)
    // index down is eta reg
    // element is kalman layer, where 7 is invalid

    // If stub with given GP encoded layer ID can have different KF layer ID depending on whether it
    // is barrel or endcap, then in layerMap, the the barrel case is assumed.
    // The endcap case is fixed by hand later in this function.

    const unsigned int nEta = 16;
    const unsigned int nGPlayID = 7;

    if (nEta != numEtaRegions_)
      throw cms::Exception("LogicError")
          << "ERROR KFbase::getKalmanLayer hardwired value of nEta differs from NumEtaRegions cfg param";

    // In cases where identical GP encoded layer ID present in this sector from both barrel & endcap, this array filled considering barrel. The endcap is fixed by subsequent code.

    constexpr unsigned layerMap[nEta / 2][nGPlayID + 1] = {
        {7, 0, 1, 5, 4, 3, 7, 2},  // B1 B2 B3 B4 B5 B6 -- current FW
        {7, 0, 1, 5, 4, 3, 7, 2},  // B1 B2 B3 B4 B5 B6
        {7, 0, 1, 5, 4, 3, 7, 2},  // B1 B2 B3 B4 B5 B6
        {7, 0, 1, 5, 4, 3, 7, 2},  // B1 B2 B3 B4 B5 B6
        {7, 0, 1, 5, 4, 3, 7, 2},  // B1 B2 B3 B4(/D3) B5(/D2) B6(/D1)
        {7, 0, 1, 3, 4, 2, 6, 2},  // B1 B2 B3(/D5)+B4(/D3) D1 D2 X D4
        {7, 0, 1, 1, 2, 3, 4, 5},  // B1 B2+D1 D2 D3 D5 D6
        {7, 0, 7, 1, 2, 3, 4, 5},  // B1 D1 D2 D3 D4 D5
    };

    unsigned int kfEtaReg;  // KF VHDL eta sector def: small in barrel & large in endcap.
    if (iEtaReg < numEtaRegions_ / 2) {
      kfEtaReg = numEtaRegions_ / 2 - 1 - iEtaReg;
    } else {
      kfEtaReg = iEtaReg - numEtaRegions_ / 2;
    }

    unsigned int kalmanLay = layerMap[kfEtaReg][layerIDreduced];

    // Fixes to layermap when "maybe layer" used
    if (settings_->kfUseMaybeLayers()) {
      switch (kfEtaReg) {
        case 5:  //case 5: B1 B2 (B3+B4)* D1 D2 D3+D4 D5+D6  -- B3 is combined with B4 and is flagged as "maybe layer"
          if (layerIDreduced == 6) {
            kalmanLay = 5;
          }
          break;
        case 6:  //case 6: B1* B2* D1 D2 D3 D4 D5 -- B1 and B2 are flagged as "maybe layer"
          if (layerIDreduced > 2) {
            kalmanLay++;
          }
          break;
        default:
          break;
      }
    }

    // Fixes to endcap stubs, for cases where identical GP encoded layer ID present in this sector from both barrel & endcap.

    if (not barrel) {
      switch (kfEtaReg) {
        case 4:  // B1 B2 B3 B4 B5/D1 B6/D2 D3
          if (layerIDreduced == 3) {
            kalmanLay = 4;
          } else if (layerIDreduced == 4) {
            kalmanLay = 5;
          } else if (layerIDreduced == 5) {
            kalmanLay = 6;
          }
          break;
          //case 5:  // B1 B2 B3+B4 D1 D2 D3 D4/D5
        case 5:  // B1 B2 B3 D1+B4 D2 D3 D4/D5
          if (layerIDreduced == 5) {
            kalmanLay = 5;
          } else if (layerIDreduced == 7) {
            kalmanLay = 6;
          }
          break;
        default:
          break;
      }
    }

    /*
  // Fix cases where a barrel layer only partially crosses the eta sector.
  // (Logically should work, but actually reduces efficiency -- INVESTIGATE).

  const float barrelHalfLength = 120.;
  const float barrel4Radius = 68.8;
  const float barrel5Radius = 86.1;
  
  if ( not barrel) {
    switch ( kfEtaReg ) {
    case 4:
      if (layerIDreduced==3) {  // D1
        float disk1_rCut = barrel5Radius*(std::abs(z)/barrelHalfLength); 
        if (r > disk1_rCut) kalmanLay++;
      }
      break;
    case 5:
      if (layerIDreduced==3) { // D1
        float disk1_rCut = barrel4Radius*(std::abs(z)/barrelHalfLength); 
        if (r > disk1_rCut) kalmanLay++;
      }
      if (layerIDreduced==4) { // D2
        float disk2_rCut = barrel4Radius*(std::abs(z)/barrelHalfLength); 
        if (r > disk2_rCut) kalmanLay++;
      }
      break;
    default:
      break;
    }			
  }
  */

    return kalmanLay;
  }

  /*=== Check if particles in given eta sector are uncertain to go through the given KF layer. */
  /*=== (If so, count layer for numbers of hit layers, but not for number of skipped layers). */

  bool KFbase::kalmanAmbiguousLayer(unsigned int iEtaReg, unsigned int kfLayer) {
    // Only helps in extreme forward sector, and there not significantly.
    // UNDERSTAND IF CAN BE USED ELSEWHERE.

    const unsigned int nEta = 16;
    const unsigned int nKFlayer = 7;
    constexpr bool ambiguityMap[nEta / 2][nKFlayer] = {
        {false, false, false, false, false, false, false},
        {false, false, false, false, false, false, false},
        {false, false, false, false, false, false, false},
        {false, false, false, false, false, false, false},
        {false, false, false, false, false, false, false},
        {false, false, true, false, false, false, false},
        {true, true, false, false, false, false, false},
        {true, false, false, false, false, false, false},
    };

    unsigned int kfEtaReg;  // KF VHDL eta sector def: small in barrel & large in endcap.
    if (iEtaReg < numEtaRegions_ / 2) {
      kfEtaReg = numEtaRegions_ / 2 - 1 - iEtaReg;
    } else {
      kfEtaReg = iEtaReg - numEtaRegions_ / 2;
    }

    bool ambiguous = false;
    if (settings_->kfUseMaybeLayers())
      ambiguous = ambiguityMap[kfEtaReg][kfLayer];

    return ambiguous;
  }

  /* Adjust KF algorithm to allow for any dead tracker layers */

  set<unsigned> KFbase::kalmanDeadLayers(bool &remove2PSCut) const {
    // Kill scenarios described StubKiller.cc

    // By which Stress Test scenario (if any) are dead modules being emulated?
    const StubKiller::KillOptions killScenario = static_cast<StubKiller::KillOptions>(settings_->killScenario());
    // Should TMTT tracking be modified to reduce efficiency loss due to dead modules?
    const bool killRecover = settings_->killRecover();

    set<pair<unsigned, bool>> deadGPlayers;  // GP layer ID & boolean indicating if in barrel.

    // Range of sectors chosen to cover dead regions from StubKiller.
    if (killRecover) {
      if (killScenario == StubKiller::KillOptions::layer5) {  // barrel layer 5
        if (iEtaReg_ >= 3 && iEtaReg_ <= 7 && iPhiSec_ >= 1 && iPhiSec_ <= 5) {
          deadGPlayers.insert(pair<unsigned, bool>(4, true));
        }
      } else if (killScenario == StubKiller::KillOptions::layer1) {  // barrel layer 1
        if (iEtaReg_ <= 7 && iPhiSec_ >= 1 && iPhiSec_ <= 5) {
          deadGPlayers.insert(pair<unsigned, bool>(1, true));
        }
        remove2PSCut = true;
      } else if (killScenario == StubKiller::KillOptions::layer1layer2) {  // barrel layers 1 & 2
        if (iEtaReg_ <= 7 && iPhiSec_ >= 1 && iPhiSec_ <= 5) {
          deadGPlayers.insert(pair<unsigned, bool>(1, true));
        }
        if (iEtaReg_ >= 1 && iEtaReg_ <= 7 && iPhiSec_ >= 1 && iPhiSec_ <= 5) {
          deadGPlayers.insert(pair<unsigned, bool>(2, true));
        }
        remove2PSCut = true;
      } else if (killScenario == StubKiller::KillOptions::layer1disk1) {  // barrel layer 1 & disk 1
        if (iEtaReg_ <= 7 && iPhiSec_ >= 1 && iPhiSec_ <= 5) {
          deadGPlayers.insert(pair<unsigned, bool>(1, true));
        }
        if (iEtaReg_ <= 3 && iPhiSec_ >= 1 && iPhiSec_ <= 5) {
          deadGPlayers.insert(pair<unsigned, bool>(3, false));
        }
        remove2PSCut = true;
      }
    }

    set<unsigned> kfDeadLayers;
    for (const auto &p : deadGPlayers) {
      unsigned int layer = p.first;
      bool barrel = p.second;
      float r = 0.;  // This fails for r-dependent parts of kalmanLayer(). FIX
      float z = 999.;
      unsigned int kalmanLay = this->kalmanLayer(iEtaReg_, layer, barrel, r, z);
      kfDeadLayers.insert(kalmanLay);
    }

    return kfDeadLayers;
  }

  //=== Function to calculate approximation for tilted barrel modules (aka B) copied from Stub class.

  float KFbase::approxB(float z, float r) const {
    return settings_->bApprox_gradient() * std::abs(z) / r + settings_->bApprox_intercept();
  }

  /* Print truth particle */

  void KFbase::printTP(const TP *tp) const {
    TVectorD tpParams(5);
    bool useForAlgEff(false);
    if (tp) {
      useForAlgEff = tp->useForAlgEff();
      tpParams[QOVERPT] = tp->qOverPt();
      tpParams[PHI0] = tp->phi0();
      tpParams[Z0] = tp->z0();
      tpParams[T] = tp->tanLambda();
      tpParams[D0] = tp->d0();
    }
    std::stringstream text;
    text << std::fixed << std::setprecision(4);
    if (tp) {
      text << "  TP index = " << tp->index() << " useForAlgEff = " << useForAlgEff << " ";
      const string helixNames[5] = {"qOverPt", "phi0", "z0", "tanL", "d0"};
      for (int i = 0; i < tpParams.GetNrows(); i++) {
        text << helixNames[i] << ":" << tpParams[i] << ", ";
      }
      text << "  inv2R = " << tp->qOverPt() * settings_->invPtToInvR() * 0.5;
    } else {
      text << "  Fake";
    }
    PrintL1trk() << text.str();
  }

  /* Print tracker layers with stubs */

  void KFbase::printStubLayers(const vector<Stub *> &stubs, unsigned int iEtaReg) const {
    std::stringstream text;
    text << std::fixed << std::setprecision(4);
    if (stubs.empty())
      text << "stub layers = []\n";
    else {
      text << "stub layers = [ ";
      for (unsigned i = 0; i < stubs.size(); i++) {
        text << stubs[i]->layerId();
        if (i != stubs.size() - 1)
          text << ", ";
      }
      text << " ]   ";
      text << "KF stub layers = [ ";
      for (unsigned j = 0; j < stubs.size(); j++) {
        unsigned int kalmanLay =
            this->kalmanLayer(iEtaReg, stubs[j]->layerIdReduced(), stubs[j]->barrel(), stubs[j]->r(), stubs[j]->z());
        text << kalmanLay;
        if (j != stubs.size() - 1)
          text << ", ";
      }
      text << " ]\n";
    }
    PrintL1trk() << text.str();
  }

  /* Print a stub */

  void KFbase::printStub(const Stub *stub) const {
    std::stringstream text;
    text << std::fixed << std::setprecision(4);
    text << "stub ";
    text << "index=" << stub->index() << " ";
    text << "layerId=" << stub->layerId() << " ";
    text << "r=" << stub->r() << " ";
    text << "phi=" << stub->phi() << " ";
    text << "z=" << stub->z() << " ";
    text << "sigmaX=" << stub->sigmaPerp() << " ";
    text << "sigmaZ=" << stub->sigmaPar() << " ";
    text << "TPids=";
    std::set<const TP *> tps = stub->assocTPs();
    for (auto tp : tps)
      text << tp->index() << ",";
    PrintL1trk() << text.str();
  }

  /* Print all stubs */

  void KFbase::printStubs(const vector<Stub *> &stubs) const {
    for (auto &stub : stubs) {
      printStub(stub);
    }
  }

}  // namespace tmtt
