#include "RecoTracker/MkFitCore/src/MiniPropagators.h"
#include "vdt/atan2.h"
#include "vdt/tan.h"
#include "vdt/sincos.h"
#include "vdt/sqrt.h"

namespace mkfit::mini_propagators {

  State::State(const MPlexLV& par, int ti) {
    x = par.constAt(ti, 0, 0);
    y = par.constAt(ti, 1, 0);
    z = par.constAt(ti, 2, 0);
    const float pt = 1.0f / par.constAt(ti, 3, 0);
    px = pt * std::cos(par.constAt(ti, 4, 0));
    py = pt * std::sin(par.constAt(ti, 4, 0));
    pz = pt / std::tan(par.constAt(ti, 5, 0));
  }

  bool InitialState::propagate_to_r(PropAlgo_e algo, float R, State& c, bool update_momentum) const {
    switch (algo) {
      case PA_Line: {
      }
      case PA_Quadratic: {
      }

      case PA_Exact: {
        // Momentum is always updated -- used as temporary for stepping.
        const float k = 1.0f / inv_k;

        const float curv = 0.5f * inv_k * inv_pt;
        const float oo_curv = 1.0f / curv;  // 2 * radius of curvature
        const float lambda = pz * inv_pt;

        float D = 0;

        c = *this;
        c.dalpha = 0;
        for (int i = 0; i < Config::Niter; ++i) {
          // compute tangental and ideal distance for the current iteration.
          // 3-rd order asin for symmetric incidence (shortest arc lenght).
          float r0 = std::hypot(c.x, c.y);
          float td = (R - r0) * curv;
          float id = oo_curv * td * (1.0f + 0.16666666f * td * td);
          // This would be for line approximation:
          // float id = R - r0;
          D += id;

          //printf("%-3d r0=%f R-r0=%f td=%f id=%f id_line=%f delta_id=%g\n",
          //       i, r0, R-r0, td, id, R - r0, id - (R-r0));

          float alpha = id * inv_pt * inv_k;
          float sina, cosa;
          vdt::fast_sincosf(alpha, sina, cosa);

          // update parameters
          c.dalpha += alpha;
          c.x += k * (c.px * sina - c.py * (1.0f - cosa));
          c.y += k * (c.py * sina + c.px * (1.0f - cosa));

          const float o_px = c.px;  // copy before overwriting
          c.px = c.px * cosa - c.py * sina;
          c.py = c.py * cosa + o_px * sina;
        }

        c.z += lambda * D;
      }
    }
    // should have some epsilon constant / member? relative vs. abs?
    c.fail_flag = std::abs(std::hypot(c.x, c.y) - R) < 0.1f ? 0 : 1;
    return c.fail_flag;
  }

  bool InitialState::propagate_to_z(PropAlgo_e algo, float Z, State& c, bool update_momentum) const {
    switch (algo) {
      case PA_Line: {
      }
      case PA_Quadratic: {
      }

      case PA_Exact: {
        const float k = 1.0f / inv_k;

        const float dz = Z - z;
        const float alpha = dz * inv_k / pz;

        float sina, cosa;
        vdt::fast_sincosf(alpha, sina, cosa);

        c.dalpha = alpha;
        c.x = x + k * (px * sina - py * (1.0f - cosa));
        c.y = y + k * (py * sina + px * (1.0f - cosa));
        c.z = Z;

        if (update_momentum) {
          c.px = px * cosa - py * sina;
          c.py = py * cosa + px * sina;
          c.pz = pz;
        }
      } break;
    }
    c.fail_flag = 0;
    return c.fail_flag;
  }

  //===========================================================================
  // Vectorized version
  //===========================================================================

  MPF fast_atan2(const MPF& y, const MPF& x) {
    MPF t;
    for (int i = 0; i < y.kTotSize; ++i) {
      t[i] = vdt::fast_atan2f(y[i], x[i]);
    }
    return t;
  }

  MPF fast_tan(const MPF& a) {
    MPF t;
    for (int i = 0; i < a.kTotSize; ++i) {
      t[i] = vdt::fast_tanf(a[i]);
    }
    return t;
  }

  void fast_sincos(const MPF& a, MPF& s, MPF& c) {
    for (int i = 0; i < a.kTotSize; ++i) {
      vdt::fast_sincosf(a[i], s[i], c[i]);
    }
  }

  StatePlex::StatePlex(const MPlexLV& par) {
    x = par.ReduceFixedIJ(0, 0);
    y = par.ReduceFixedIJ(1, 0);
    z = par.ReduceFixedIJ(2, 0);
    const MPF pt = 1.0f / par.ReduceFixedIJ(3, 0);
    fast_sincos(par.ReduceFixedIJ(4, 0), py, px);
    px *= pt;
    py *= pt;
    pz = pt / fast_tan(par.ReduceFixedIJ(5, 0));
  }

  // propagate to radius; returns number of failed propagations
  int InitialStatePlex::propagate_to_r(
      PropAlgo_e algo, const MPF& R, StatePlex& c, bool update_momentum, int N_proc) const {
    switch (algo) {
      case PA_Line: {
      }
      case PA_Quadratic: {
      }

      case PA_Exact: {
        // Momentum is always updated -- used as temporary for stepping.
        const MPF k = 1.0f / inv_k;

        const MPF curv = 0.5f * inv_k * inv_pt;
        const MPF oo_curv = 1.0f / curv;  // 2 * radius of curvature
        const MPF lambda = pz * inv_pt;

        MPF D = 0;

        c = *this;
        c.dalpha = 0;
        for (int i = 0; i < Config::Niter; ++i) {
          // compute tangental and ideal distance for the current iteration.
          // 3-rd order asin for symmetric incidence (shortest arc lenght).
          MPF r0 = Matriplex::hypot(c.x, c.y);
          MPF td = (R - r0) * curv;
          MPF id = oo_curv * td * (1.0f + 0.16666666f * td * td);
          // This would be for line approximation:
          // float id = R - r0;
          D += id;

          //printf("%-3d r0=%f R-r0=%f td=%f id=%f id_line=%f delta_id=%g\n",
          //       i, r0, R-r0, td, id, R - r0, id - (R-r0));

          MPF alpha = id * inv_pt * inv_k;

          MPF sina, cosa;
          fast_sincos(alpha, sina, cosa);

          // update parameters
          c.dalpha += alpha;
          c.x += k * (c.px * sina - c.py * (1.0f - cosa));
          c.y += k * (c.py * sina + c.px * (1.0f - cosa));

          MPF o_px = c.px;  // copy before overwriting
          c.px = c.px * cosa - c.py * sina;
          c.py = c.py * cosa + o_px * sina;
        }

        c.z += lambda * D;
      }
    }

    // should have some epsilon constant / member? relative vs. abs?
    MPF r = Matriplex::hypot(c.x, c.y);
    c.fail_flag = 0;
    int n_fail = 0;
    for (int i = 0; i < N_proc; ++i) {
      if (std::abs(R[i] - r[i]) > 0.1f) {
        c.fail_flag[i] = 1;
        ++n_fail;
      }
    }
    return n_fail;
  }

  int InitialStatePlex::propagate_to_z(
      PropAlgo_e algo, const MPF& Z, StatePlex& c, bool update_momentum, int N_proc) const {
    switch (algo) {
      case PA_Line: {
      }
      case PA_Quadratic: {
      }

      case PA_Exact: {
        MPF k = 1.0f / inv_k;

        MPF dz = Z - z;
        MPF alpha = dz * inv_k / pz;

        MPF sina, cosa;
        fast_sincos(alpha, sina, cosa);

        c.dalpha = alpha;
        c.x = x + k * (px * sina - py * (1.0f - cosa));
        c.y = y + k * (py * sina + px * (1.0f - cosa));
        c.z = Z;

        if (update_momentum) {
          c.px = px * cosa - py * sina;
          c.py = py * cosa + px * sina;
          c.pz = pz;
        }
      } break;
    }
    c.fail_flag = 0;
    return 0;
  }

}  // namespace mkfit::mini_propagators
