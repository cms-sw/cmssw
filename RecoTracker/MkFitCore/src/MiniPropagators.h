#ifndef RecoTracker_MkFitCore_src_MiniPropagators_h
#define RecoTracker_MkFitCore_src_MiniPropagators_h

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/MatrixSTypes.h"
#include "Matrix.h"

namespace mkfit {
  struct ModuleInfo;
}

namespace mkfit::mini_propagators {

  enum PropAlgo_e { PA_Line, PA_Quadratic, PA_Exact };

  struct State {
    float x, y, z;
    float px, py, pz;
    float dalpha;
    int fail_flag;

    State() = default;
    State(const MPlexLV& par, int ti);
  };

  struct InitialState : public State {
    float inv_pt, inv_k;
    float theta;

    InitialState(const MPlexLV& par, const MPlexQI& chg, int ti)
        : InitialState(State(par, ti), chg.constAt(ti, 0, 0), par.constAt(ti, 3, 0), par.constAt(ti, 5, 0)) {}

    InitialState(State s, short charge, float ipt, float tht, float bf = Config::Bfield)
        : State(s), inv_pt(ipt), theta(tht) {
      inv_k = ((charge < 0) ? 0.01f : -0.01f) * Const::sol * bf;
    }

    bool propagate_to_r(PropAlgo_e algo, float R, State& c, bool update_momentum) const;
    bool propagate_to_z(PropAlgo_e algo, float Z, State& c, bool update_momentum) const;

    bool propagate_to_plane(PropAlgo_e algo, const ModuleInfo& mi, State& c, bool update_momentum) const;
  };

  //-----------------------------------------------------------
  // Vectorized version
  //-----------------------------------------------------------

  using MPF = MPlexQF;
  using MPI = MPlexQI;

  MPF fast_atan2(const MPF& y, const MPF& x);
  MPF fast_tan(const MPF& a);
  void fast_sincos(const MPF& a, MPF& s, MPF& c);

  struct StatePlex {
    MPF x, y, z;
    MPF px, py, pz;
    MPF dalpha;
    MPI fail_flag{0};

    StatePlex() = default;
    StatePlex(const MPlexLV& par);
  };

  struct InitialStatePlex : public StatePlex {
    MPF inv_pt, inv_k;
    MPF theta;

    InitialStatePlex(const MPlexLV& par, const MPI& chg)
        : InitialStatePlex(StatePlex(par), chg, par.ReduceFixedIJ(3, 0), par.ReduceFixedIJ(5, 0)) {}

    InitialStatePlex(StatePlex s, MPI charge, MPF ipt, MPF tht, float bf = Config::Bfield)
        : StatePlex(s), inv_pt(ipt), theta(tht) {
      for (int i = 0; i < inv_k.kTotSize; ++i) {
        inv_k[i] = ((charge[i] < 0) ? 0.01f : -0.01f) * Const::sol * bf;
      }
    }

    int propagate_to_r(PropAlgo_e algo, const MPF& R, StatePlex& c, bool update_momentum, int N_proc = NN) const;
    int propagate_to_z(PropAlgo_e algo, const MPF& Z, StatePlex& c, bool update_momentum, int N_proc = NN) const;
  };

};  // namespace mkfit::mini_propagators

#endif
