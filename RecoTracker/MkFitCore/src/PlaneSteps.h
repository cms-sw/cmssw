#ifndef RecoTracker_MkFitCore_src_PlaneSteps_h
#define RecoTracker_MkFitCore_src_PlaneSteps_h

#include "Matrix.h"
#include "RecoTracker/MkFitCore/interface/FunctionTypes.h"

namespace mkfit {

  class PropagationFlags;
  class TrackerInfo;

  // Propagation to a plane and the Kalman operation on it, written as a sequence of steps.
  //
  // A step reads the structs it is given and fills one output.  The structs below are what is
  // known at each point of the sequence, so a step cannot be called before its inputs exist,
  // and the state a quantity is evaluated at is always a named argument.
  //
  // Inputs the caller already holds are passed as views (structs of const references); outputs
  // own their storage.  The steps are defined in PropagationMPlexPlane.cc and KalmanUtilsMPlex.cc;
  // the public functions declared in PropagationMPlex.h and KalmanUtilsMPlex.h are sequences of
  // them, and other code can compose its own.

  namespace plane {

    // Track parameters the caller holds, with the charge and the number of used lanes.  The
    // covariance is passed explicitly to the steps that read it.
    struct TrackRef {
      const MPlexLV& par;
      const MPlexQI& chg;
      const int n_proc;
    };

    // The plane: a point on it and its normal.
    struct PlaneRef {
      const MPlexHV& pnt;
      const MPlexHV& nrm;
    };

    // The field used for one propagation step: Bz [T] per lane and the curvature factor
    // kinv = -q * sol/100 * Bz.
    struct FieldAt {
      MPlexQF b;
      MPlexQF kinv;
    };

    // Trigonometry of the start state: sin and cos of the momentum's phi and theta.  Computed once per
    // start state and read by every step that starts from it.
    struct StartTrig {
      MPlexQF sinP, cosP;
      MPlexQF sinT, cosT;
    };

    // The path length to the plane while it is being solved: s, and the start state's straight-line
    // solution, which the closing step falls back to.  s starts at zero on every lane, so lanes beyond
    // n_proc read zero.
    struct PathSolve {
      MPlexQF s{0.0f};
      MPlexQF s_line;
    };

    // Material to apply at the destination of a step: radiation length and Bethe-Bloch xi at
    // normal incidence, and the sign of the energy loss.
    struct MaterialAt {
      MPlexQF radl;
      MPlexQF xi;
      MPlexQF sign;
    };

    // The predicted state in the local frame of the plane, as the Kalman operation uses it:
    // the rotation into the frame, the local position, the local parameters
    // (q/p, dx/dz, dy/dz, x, y), the sign of the local z momentum and the local covariance.
    struct LocalPred {
      MPlexHH rot;
      MPlex2V xlo;
      MPlex5V lp;
      MPlexQI pz_sign;
      MPlex5S err;
    };

    // The measurement in the local frame of the plane: position and covariance.
    struct LocalMeas {
      MPlex2V par;
      MPlex2S err;
    };

    // Residual of the measurement against the prediction, the INVERSE of its covariance, and the
    // determinant of the covariance (before the inversion).
    struct Residual {
      MPlex2V r;
      MPlex2S s_inv;
      MPlexQF det;
    };

    // The updated state in the local frame: parameters and covariance.
    struct LocalUpd {
      MPlex5V lp;
      MPlex5S err;
    };

    // ---- Propagation to a plane (PropagationMPlexPlane.cc)

    // The field at the start state: parametrised (pf.use_param_b_field) or constant.
    void field_at_start(const TrackRef& in, const PropagationFlags& pf, FieldAt& f);

    // The trigonometry of the start state.
    void start_trig(const TrackRef& in, StartTrig& t);

    // The path length from the start state to the plane in the field f: a first solve, refinements
    // from the state the current solution reaches, and the straight line where the helix solution is
    // not finite.  path_solve() does the three with Config::nSStepsInProp2Plane - 1 refinements.
    void path_init(const TrackRef& in, const StartTrig& t, const PlaneRef& pl, const FieldAt& f, PathSolve& p);
    void path_refine(const TrackRef& in, const StartTrig& t, const PlaneRef& pl, const FieldAt& f, PathSolve& p);
    void path_close(const TrackRef& in, PathSolve& p);
    void path_solve(const TrackRef& in, const StartTrig& t, const PlaneRef& pl, const FieldAt& f, PathSolve& p);

    // The path length from the transverse one, when the caller already knows the crossing.
    void path_from_perp(const TrackRef& in, const StartTrig& t, const MPlexQF& sPerp, MPlexQF& s);

    // Parameters after a helix step of path length s from the start state, in the field f.
    void drift(const TrackRef& in, const StartTrig& t, const FieldAt& f, const MPlexQF& s, MPlexLV& outPar);

    // Transport Jacobian of the step from the start state to outPar, in the field f.
    void jacobian(
        const TrackRef& in, const StartTrig& t, const MPlexLV& outPar, const FieldAt& f, const MPlexQF& s, MPlexLL& J);

    // Covariance at the destination: J C J^T.
    void transport_cov(const MPlexLL& J, const MPlexLS& inErr, MPlexLS& outErr);

    // Material at the destination: from the (|z|, r) grid at par, or given per lane.  None on the lanes
    // masked by noMatEffPtr and beyond N_proc.
    void material_grid(
        const TrackerInfo& tinfo, const MPlexLV& par, const MPlexQI* noMatEffPtr, const int N_proc, MaterialAt& m);
    void material_given(
        const MPlexQF& radl, const MPlexQF& xi, const MPlexQI* noMatEffPtr, const int N_proc, MaterialAt& m);

    // Sign of the energy loss: from the sign of the path length, or from the pass (outward loses).
    void eloss_sign_from_path(const MPlexQF& s, const MPlexQI* noMatEffPtr, const int N_proc, MaterialAt& m);
    void eloss_sign_of_pass(const bool outward, const MPlexQI* noMatEffPtr, const int N_proc, MaterialAt& m);

    // Multiple scattering and energy loss of the material m on the plane with normal plNrm.
    void apply_material(const MaterialAt& m, const MPlexHV& plNrm, MPlexLS& err, MPlexLV& par, const int N_proc);

    // Phi into [-pi, pi), and the start state (in, inErr) restored on lanes whose propagation failed.
    void finish(const TrackRef& in, const MPlexLS& inErr, const MPlexQI& failFlag, MPlexLV& par, MPlexLS& err);

    // ---- Kalman operation on a plane (KalmanUtilsMPlex.cc)

    // sol/100 * Bz at the predicted state: the field both local-frame Jacobians are evaluated in.
    void local_field(const TrackRef& pred, const bool use_param_b_field, MPlexQF& bFld);

    // The predicted state and its covariance psErr in the local frame of the plane (pl.nrm, plDir).
    void to_local(const TrackRef& pred,
                  const MPlexLS& psErr,
                  const PlaneRef& pl,
                  const MPlexHV& plDir,
                  const MPlexQF& bFld,
                  LocalPred& L);

    // The measurement (msPar, msErr) in the local frame of the plane.
    void measurement(const LocalPred& L, const MPlexHV& msPar, const MPlexHS& msErr, const MPlexHV& plPnt, LocalMeas& M);

    // The CPE's local position and error replace the measurement on the lanes where doCPE holds a hit
    // index and the CPE succeeds.
    void measurement_cpe(const MPlexQI& doCPE, const cpe_func& cpe_corr_func, const LocalPred& L, LocalMeas& M);

    // Residual of the measurement against the prediction, its covariance inverted, and its determinant.
    void residual(const LocalPred& L, const LocalMeas& M, Residual& R);

    // chi2 of the residual.
    void chi2(const Residual& R, MPlexQF& outChi2);

    // chi2 of the residual with the track part of its covariance scaled by trk_scale2.
    void chi2_scaled(const LocalPred& L, const LocalMeas& M, const Residual& R, const float trk_scale2, MPlexQF& out);

    // Kalman gain and the updated local state.
    void update(const LocalPred& L, const Residual& R, LocalUpd& U);

    // The updated state back in global coordinates (CCS): parameters and covariance.
    void to_global(const LocalUpd& U,
                   const LocalPred& L,
                   const PlaneRef& pl,
                   const MPlexQI& inChg,
                   const MPlexQF& bFld,
                   MPlexLV& outPar,
                   MPlexLS& outErr);

  }  // namespace plane

}  // namespace mkfit

#endif
