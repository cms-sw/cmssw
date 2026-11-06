#ifndef RecoTracker_MkFitCore_src_FinalFit_h
#define RecoTracker_MkFitCore_src_FinalFit_h

#include "Matrix.h"
#include "RecoTracker/MkFitCore/interface/FunctionTypes.h"

namespace mkfit {

  class PropagationFlags;
  struct FinalFitFlags;

  // The final fit's propagation to the module planes and its Kalman update on them (MkFitter, called from
  // MkBuilder::fit_tracks()), composed from the steps in PlaneSteps.h.  The final fit has its own sequences so that
  // what only it does stays out of the propagation and Kalman functions that track finding uses.

  namespace final_fit {

    // What stays fixed over one pass of the final fit: the propagation flags, the final fit's own choices, the
    // direction of the pass (the forward pass runs outward), the per-lane |p| at which the scattering noise is
    // evaluated (nullptr: the running estimate), and the number of sub-steps of each propagation.
    struct Pass {
      const PropagationFlags& pflags;
      const FinalFitFlags& ffflags;
      const bool outward;
      const float* ms_ref_p = nullptr;
      const int n_sub = 1;
    };

    // The material of each lane's destination module (ModuleInfo::radl and bbxi), used in place of the material
    // grid when FinalFitFlags::material_per_module is set.
    struct ModuleMaterial {
      const MPlexQF& radl;
      const MPlexQF& bbxi;
    };

    // Optional outputs of the update, in the local frame of the module plane (q/p, dx/dz, dy/dz, x, y) as the update
    // computes them: the predicted and the updated state and the sign of the local z momentum.  Each pointer may be
    // null.  For the final fit's per-hit states.
    struct LocalStatesOut {
      MPlex5V* predPar = nullptr;
      MPlex5S* predErr = nullptr;
      MPlex5V* updPar = nullptr;
      MPlex5S* updErr = nullptr;
      MPlexQI* pzSign = nullptr;
    };

    // Propagation of (inErr, inPar) to the planes (plPnt, plNrm), with material at the destination.
    void propagate(const MPlexLS& inErr,
                   const MPlexLV& inPar,
                   const MPlexQI& inChg,
                   const MPlexHV& plPnt,
                   const MPlexHV& plNrm,
                   MPlexLS& outErr,
                   MPlexLV& outPar,
                   MPlexQI& outFailFlag,
                   const int N_proc,
                   const Pass& pass,
                   const MPlexQI* noMatEffPtr,
                   const ModuleMaterial* modMat);

    // Propagation of (inErr, inPar) to the planes in nSub sub-steps: the parameters through nSub - 1 drifts of
    // fixed path length s0/nSub (s0 = the first path-length estimate to the plane), each with the field model of a
    // full step, then onto the plane with the full solve; the covariance with the Jacobian of the whole step;
    // material at the destination.  split[n] = false keeps lane n as one step.
    void propagate_sub_steps(const MPlexLS& inErr,
                             const MPlexLV& inPar,
                             const MPlexQI& inChg,
                             const MPlexHV& plPnt,
                             const MPlexHV& plNrm,
                             MPlexLS& outErr,
                             MPlexLV& outPar,
                             MPlexQI& outFailFlag,
                             const int N_proc,
                             const Pass& pass,
                             const int nSub,
                             const bool* split,
                             const MPlexQI* noMatEffPtr,
                             const ModuleMaterial* modMat);

    // Propagation to the hits' planes, then the Kalman update with the hits and its chi2, with the updated covariance
    // in Joseph form.  A negative q/p after the update flips the charge.  With pass.n_sub > 1 the lanes not yet on
    // their plane are propagated in sub-steps.
    void propagate_update(const MPlexLS& psErr,
                          const MPlexLV& psPar,
                          MPlexQI& Chg,
                          const MPlexHS& msErr,
                          const MPlexHV& msPar,
                          const MPlexHV& plNrm,
                          const MPlexHV& plDir,
                          const MPlexHV& plPnt,
                          MPlexLS& outErr,
                          MPlexLV& outPar,
                          MPlexQI& outFailFlag,
                          MPlexQF& outChi2,
                          const int N_proc,
                          const Pass& pass,
                          const MPlexQI* noMatEffPtr,
                          const MPlexQI* doCPE,
                          cpe_func cpe_corr_func,
                          const ModuleMaterial* modMat,
                          const LocalStatesOut* localStates = nullptr);

    // Two-filter smoother on one module plane: combine two independent local estimates of the same state (the
    // forward updated and the backward predicted one): xs = xf + Cf S^-1 (xb - xf), Cs = Cf S^-1 Cb (computed in Joseph
    // form), S = Cf + Cb.
    // ok = 0 where S is not positive definite (a pivot of its Cholesky factorisation is not positive; a full test,
    // not only of the diagonal).
    void smooth_local_states(const MPlex5V& xf,
                             const MPlex5S& cf,
                             const MPlex5V& xb,
                             const MPlex5S& cb,
                             MPlex5V& xs,
                             MPlex5S& cs,
                             MPlexQI& ok,
                             const int N_proc);

  }  // namespace final_fit

}  // namespace mkfit

#endif
