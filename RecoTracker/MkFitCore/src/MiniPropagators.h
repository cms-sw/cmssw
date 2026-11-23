#ifndef RecoTracker_MkFitCore_src_MiniPropagators_h
#define RecoTracker_MkFitCore_src_MiniPropagators_h

#include "RecoTracker/MkFitCore/src/Matrix.h"
#include "RecoTracker/MkFitCore/interface/TrackState.h"

namespace mkfit::mini_propagators {

  enum PropAlgo_e { PA_Line, PA_Quadratic, PA_Exact };

  // Guard band [cm] kept inside the radii the trajectory can reach, where the
  // crossing becomes ill-conditioned (dr/ds -> 0). Provisional, see
  // doc/MkFinderV2p2-DesignNotes.md, "Reachability".
  constexpr float kReachMargin = 0.1f;

  // How close to the target radius propagate_to_r() must land to count as
  // converged. With the closed form only round-off should trip it.
  constexpr float kReachTolerance = 0.1f;

  struct StatePlex;
  struct InitialStatePlex;

  struct State {
    float x, y, z;
    float px, py, pz;
    float dalpha;
    int fail_flag;

    State(float x = 0,
          float y = 0,
          float z = 0,
          float px = 0,
          float py = 0,
          float pz = 0,
          float dalpha = 0,
          int fail_flag = 0)
        : x(x), y(y), z(z), px(px), py(py), pz(pz), dalpha(dalpha), fail_flag(fail_flag) {}
    State(const MPlexLV& par, int ti);
    State(const StatePlex& sp, int i);
  };

  struct InitialState : public State {
    float inv_pt, inv_k;
    float theta;

    InitialState() : State(), inv_pt(0), inv_k(0), theta(0) {}

    InitialState(const MPlexLV& par, const MPlexQI& chg, int ti, float bf = Config::Bfield)
        : InitialState(State(par, ti), chg.constAt(ti, 0, 0), par.constAt(ti, 3, 0), par.constAt(ti, 5, 0), bf) {}

    InitialState(const State& s, short charge, float ipt, float tht, float bf = Config::Bfield)
        : State(s), inv_pt(ipt), theta(tht) {
      inv_k = ((charge < 0) ? 0.01f : -0.01f) * Const::sol * bf;
    }

    InitialState(const StatePlex& sp, int i, const InitialStatePlex& isp, int j);

    bool propagate_to_r(PropAlgo_e algo, float R, State& c, bool update_momentum) const;
    bool propagate_to_z(PropAlgo_e algo, float Z, State& c, bool update_momentum) const;

    bool propagate_to_plane(
        PropAlgo_e algo, const SVector3& pos, const SVector3& zdir, State& c, bool update_momentum) const;
  };

  //-----------------------------------------------------------
  // Vectorized version
  //-----------------------------------------------------------

  using MPF = MPlexQF;
  using MPI = MPlexQI;
  using MP3V = MPlex3V;
  using MP4V = MPlex4V;

  struct StatePlex {
    MPF x, y, z;
    MPF px, py, pz;
    MPF dalpha{0.0f};
    MPI fail_flag{0};

    StatePlex() = default;
    StatePlex(const MPlexLV& par);

    void copyIn(int dst_slot, const State& src) {
      x[dst_slot] = src.x;
      y[dst_slot] = src.y;
      z[dst_slot] = src.z;
      px[dst_slot] = src.px;
      py[dst_slot] = src.py;
      pz[dst_slot] = src.pz;
      dalpha[dst_slot] = src.dalpha;
      fail_flag[dst_slot] = src.fail_flag;
    }

    void copyIn(int dst_slot, const StatePlex& src, int src_slot) {
      x[dst_slot] = src.x[src_slot];
      y[dst_slot] = src.y[src_slot];
      z[dst_slot] = src.z[src_slot];
      px[dst_slot] = src.px[src_slot];
      py[dst_slot] = src.py[src_slot];
      pz[dst_slot] = src.pz[src_slot];
      dalpha[dst_slot] = src.dalpha[src_slot];
      fail_flag[dst_slot] = src.fail_flag[src_slot];
    }

    State state(int i) const { return {x[i], y[i], z[i], px[i], py[i], pz[i], dalpha[i], fail_flag[i]}; }
  };

  struct InitialStatePlex : public StatePlex {
    MPF inv_pt, inv_k;
    MPF theta;

    InitialStatePlex() = default;

    InitialStatePlex(const MPlexLV& par, const MPI& chg, float bf = Config::Bfield)
        : InitialStatePlex(StatePlex(par), chg, par.ReduceFixedIJ(3, 0), par.ReduceFixedIJ(5, 0), bf) {}

    // The same with a field value per lane, e.g. Config::bFieldFromZR() at each state.
    InitialStatePlex(const MPlexLV& par, const MPI& chg, const MPF& bf)
        : InitialStatePlex(StatePlex(par), chg, par.ReduceFixedIJ(3, 0), par.ReduceFixedIJ(5, 0), bf) {}

    InitialStatePlex(const StatePlex& sp, MPI charge, MPF ipt, MPF tht, float bf = Config::Bfield)
        : StatePlex(sp), inv_pt(ipt), theta(tht) {
      for (int i = 0; i < inv_k.kTotSize; ++i) {
        inv_k[i] = ((charge[i] < 0) ? 0.01f : -0.01f) * Const::sol * bf;
      }
    }

    InitialStatePlex(const StatePlex& sp, MPI charge, MPF ipt, MPF tht, const MPF& bf)
        : StatePlex(sp), inv_pt(ipt), theta(tht) {
      for (int i = 0; i < inv_k.kTotSize; ++i) {
        inv_k[i] = ((charge[i] < 0) ? 0.01f : -0.01f) * Const::sol * bf[i];
      }
    }

    InitialStatePlex(const StatePlex& sp, const InitialStatePlex& isp)
        : StatePlex(sp), inv_pt(isp.inv_pt), inv_k(isp.inv_k), theta(isp.theta) {}

    using StatePlex::operator=;

    void copyIn(int dst_slot, const InitialState& src) {
      StatePlex::copyIn(dst_slot, src);
      inv_pt[dst_slot] = src.inv_pt;
      inv_k[dst_slot] = src.inv_k;
      theta[dst_slot] = src.theta;
    }

    void copyIn(int dst_slot, const InitialStatePlex& src, int src_slot) {
      StatePlex::copyIn(dst_slot, src, src_slot);
      inv_pt[dst_slot] = src.inv_pt[src_slot];
      inv_k[dst_slot] = src.inv_k[src_slot];
      theta[dst_slot] = src.theta[src_slot];
    }

    // Once slots are filled, to be followed by a call to init_momentum_vec_and_k()
    void copyIn_partial_track_state(int dst_slot, const mkfit::TrackState& src) {
      x[dst_slot] = src.x();
      y[dst_slot] = src.y();
      z[dst_slot] = src.z();
      inv_pt[dst_slot] = src.invpT();
      theta[dst_slot] = src.theta();
    }

    void init_momentum_vec_and_k(const MPF& phi, const MPI& chg, float bf = Config::Bfield);
    // The same with a field value per lane, e.g. Config::bFieldFromZR() at each state.
    void init_momentum_vec_and_k(const MPF& phi, const MPI& chg, const MPF& bf);

    int propagate_to_r(PropAlgo_e algo, const MPF& R, StatePlex& c, bool update_momentum, int N_proc = NN) const;
    int propagate_to_z(PropAlgo_e algo, const MPF& Z, StatePlex& c, bool update_momentum, int N_proc = NN) const;

    int propagate_to_plane(PropAlgo_e algo, const MP3V& pos, const MP3V& zdir, StatePlex& c, bool update_momentum) const;
  };

  // Projecting constructors from plexes to regular state objects

  inline State::State(const StatePlex& sp, int i) : State(sp.state(i)) {}

  inline InitialState::InitialState(const StatePlex& sp, int i, const InitialStatePlex& isp, int j)
      : State(sp.state(i)), inv_pt(isp.inv_pt[j]), inv_k(isp.inv_k[j]), theta(isp.theta[j]) {}

  //-----------------------------------------------------------
  // Hermite interpolation
  //-----------------------------------------------------------

  // Hermite3D -- cubic model of the helix, parametrized by t.
  //
  // In both modes below t is LINEAR in the helix turning angle alpha, spanning
  // dalpha over t in [0,1]. Everything downstream depends on that and on nothing
  // else: Hermite3DOnPlane, the momentum returned by evaluate(), the linear
  // dalpha interpolation in MkFinderV2p2, and path_length() below.
  //
  // The helix, exactly, in the turning angle alpha (k = 1/inv_k, p in GeV):
  //   x(a) = x0 + k (px sin a - py (1 - cos a))
  //   y(a) = y0 + k (py sin a + px (1 - cos a))
  //   z(a) = z0 + k pz a
  //   p(a) = p rotated by a about z;  pz constant
  // so  dr/da = k p(a)  exactly, which is where m_Hderfac comes from and why
  // z is reproduced exactly (it is linear in a) while x, y are cubic-accurate.
  struct Hermite3D {
    // Hermite p(3) for trajectory approximation
    MP4V m_Hx, m_Hy, m_Hz;  // Hermite p(3) coefficients for x, y, z
    MPF m_Hderfac;          // 1 / (k * dalpha) -- turns dr/dt back into momentum

    Hermite3D() = default;
    explicit Hermite3D(float val) : m_Hx(val), m_Hy(val), m_Hz(val), m_Hderfac(val) {}

    void copyIn(int dst_slot, const Hermite3D& src, int src_slot) {
      m_Hx.copyIn(dst_slot, src.m_Hx, src_slot);
      m_Hy.copyIn(dst_slot, src.m_Hy, src_slot);
      m_Hz.copyIn(dst_slot, src.m_Hz, src_slot);
      m_Hderfac.copyIn(dst_slot, src.m_Hderfac, src_slot);
    }

    static void hermite_1d(const MPF& x1, const MPF& p1, const MPF& x2, const MPF& p2, const MPF& derfac, MP4V& h);

    // Two-point / interpolating mode: t = 0 at sp1, t = 1 at sp2, matching
    // position and momentum at both. Truncation error ~ R_c * dalpha^4 / 384.
    void calculate_coeffs(const StatePlex& sp1, const StatePlex& sp2, const MPF& inv_k);

    // One-point / extrapolating mode: Taylor expansion of the helix about sp,
    // with t = 1 at sp.dalpha + dalpha. Needs no second propagation -- the
    // derivatives are known analytically from the helix model:
    //   r'   = k ( px,  py, pz)
    //   r''  = k (-py,  px, 0 )
    //   r''' = k (-px, -py, 0 )
    // Truncation error ~ R_c * dalpha^4 / 24, i.e. 16x the two-point mode, so
    // half the usable span at equal accuracy. Usable transverse path for 10 um
    // is s ~ (24 eps)^1/4 * R_c^3/4: about 11 cm at 1 GeV, 63 cm at 10 GeV --
    // ample for re-expanding inside a layer after a Kalman update, the intended
    // use; not called in the search yet. Cheaper than the two-point mode: no
    // trig, no propagation.
    void calculate_coeffs(const StatePlex& sp, const MPF& inv_k, const MPF& dalpha);

    void evaluate(const MPF& t, MPF& x, MPF& y, MPF& z) const;
    void evaluate(const MPF& t, MPF& x, MPF& y, MPF& z, MPF& dx, MPF& dy, MPF& dz) const;
    void evaluate(const MPF& t, StatePlex& sp) const;

    // Signed path length from the expansion point to t, positive along
    // increasing t. EXACT in both modes -- arc length is linear in the helix
    // angle -- so this is the length the Kalman propagation would take (not
    // called yet), not an approximation of it. Pass |p| for the 3D length, pT for the transverse
    // one; k cancels, which is why only m_Hderfac is needed.
    //   s = k * |p| * alpha,  alpha = dalpha * t,  m_Hderfac = 1 / (k * dalpha)
    MPF path_length(const MPF& t, const MPF& p_mag) const { return p_mag * t / m_Hderfac; }
  };

  struct Hermite3DOnPlane {
    MP4V m_D;  // p(3) for distance to plane
    MPF m_T;   // t at solution

    void init_coeffs(const Hermite3D& h, const MP3V& pos, const MP3V& zdir);

    void evaluate(const MPF& t, MPF& d) const;
    void evaluate(const MPF& t, MPF& d, MPF& dddt) const;

    void solve();
  };

}  // namespace mkfit::mini_propagators

#endif
