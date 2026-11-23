#include "RecoTracker/MkFitCore/src/MiniPropagators.h"
#include "RecoTracker/MkFitCore/interface/MathInlineCore.h"
#include <vdt/atan2.h>

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
    // Scalar mirror of InitialStatePlex::propagate_to_r(). The derivation, the
    // precision argument for dc2_m_Rc2, and the meaning of the fail_flag values
    // are documented there -- keep the two in step.
    float R_eff = R;
    bool unreachable = false;

    switch (algo) {
      case PA_Line: {
      }
      case PA_Quadratic: {
      }

      case PA_Exact: {
        const float k = 1.0f / inv_k;

        c = *this;

        const float cx = x - k * py;
        const float cy = y + k * px;
        const float d_c = hipo(cx, cy);
        const float R_c = std::abs(k) * hipo(px, py);
        const float dc2_m_Rc2 = x * x + y * y - 2.0f * k * (x * py - y * px);

        float r_lo = std::abs(d_c - R_c) + kReachMargin;
        float r_hi = d_c + R_c - kReachMargin;
        if (r_lo > r_hi)
          r_lo = r_hi = 0.5f * (std::abs(d_c - R_c) + d_c + R_c);
        if (R < r_lo) {
          R_eff = r_lo;
          unreachable = true;
        } else if (R > r_hi) {
          R_eff = r_hi;
          unreachable = true;
        }

        const float R2 = R_eff * R_eff;
        const float a = (R2 + dc2_m_Rc2) / (2.0f * d_c);
        const float b2 = R2 - a * a;
        const float b = b2 > 0.0f ? std::sqrt(b2) : 0.0f;

        const float inv_dc = 1.0f / d_c;
        const float hx = cx * inv_dc, hy = cy * inv_dc;  // Chat; Chat_perp = (-hy, hx)

        const float u0x = x - cx, u0y = y - cy;
        const float inv_Rc2 = 1.0f / (R_c * R_c);

        const float p1x = a * hx - b * hy, p1y = a * hy + b * hx;
        const float p2x = a * hx + b * hy, p2y = a * hy - b * hx;
        const float u1x = p1x - cx, u1y = p1y - cy;
        const float u2x = p2x - cx, u2y = p2y - cy;
        const float cos1 = (u0x * u1x + u0y * u1y) * inv_Rc2;
        const float sin1 = (u0x * u1y - u0y * u1x) * inv_Rc2;
        const float cos2 = (u0x * u2x + u0y * u2y) * inv_Rc2;
        const float sin2 = (u0x * u2y - u0y * u2x) * inv_Rc2;
        const float alpha1 = vdt::fast_atan2f(sin1, cos1);
        const float alpha2 = vdt::fast_atan2f(sin2, cos2);

        const bool first = std::abs(alpha1) <= std::abs(alpha2);
        const float alpha = first ? alpha1 : alpha2;
        const float sina = first ? sin1 : sin2;
        const float cosa = first ? cos1 : cos2;
        c.x = first ? p1x : p2x;
        c.y = first ? p1y : p2y;

        c.z = z + k * pz * alpha;
        c.dalpha = dalpha + alpha;

        if (update_momentum) {
          const float o_px = px;
          c.px = px * cosa - py * sina;
          c.py = py * cosa + o_px * sina;
          c.pz = pz;
        }
      }
    }

    if (unreachable)
      c.fail_flag = 2;
    else
      c.fail_flag = std::abs(R_eff - hipo(c.x, c.y)) > kReachTolerance ? 1 : 0;
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

        c.dalpha = dalpha + alpha;
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

  bool InitialState::propagate_to_plane(
      PropAlgo_e algo, const SVector3& pos, const SVector3& zdir, State& c, bool update_momentum) const {
    switch (algo) {
      case PA_Line: {
        // Straight-line step along the momentum onto the plane; the momentum
        // is not changed.
        // t * p_vec intersects the plane:
        float t = (pos(0) * zdir(0) + pos(1) * zdir(1) + pos(2) * zdir(2) - x * zdir(0) - y * zdir(1) - z * zdir(2)) /
                  (px * zdir(0) + py * zdir(1) + pz * zdir(2));

        c = *this;
        c.x += t * c.px;
        c.y += t * c.py;
        c.z += t * c.pz;

        // Signed helix turning angle over the same transverse path, t pT / (k pT).
        c.dalpha += t * inv_k;
        break;
      }

      case PA_Quadratic: {
        throw std::runtime_error("Quadratic prop_to_plane not implemented");
      }

      case PA_Exact: {
        throw std::runtime_error("Exact prop_to_plane not implemented");
      }
    }
    return false;
  }

  //===========================================================================
  // Vectorized version
  //===========================================================================

  StatePlex::StatePlex(const MPlexLV& par) {
    x = par.ReduceFixedIJ(0, 0);
    y = par.ReduceFixedIJ(1, 0);
    z = par.ReduceFixedIJ(2, 0);
    const MPF pt = 1.0f / par.ReduceFixedIJ(3, 0);
    Matriplex::fast_sincos(par.ReduceFixedIJ(4, 0), py, px);
    px *= pt;
    py *= pt;
    pz = pt / Matriplex::fast_tan(par.ReduceFixedIJ(5, 0));
  }

  void InitialStatePlex::init_momentum_vec_and_k(const MPF& phi, const MPI& chg, float bf) {
    const MPF pt = 1.0f / inv_pt;
    phi.fast_sincos(py, px);
    px *= pt;
    py *= pt;
    pz = pt / Matriplex::fast_tan(theta);
    for (int i = 0; i < inv_k.kTotSize; ++i) {
      inv_k[i] = ((chg[i] < 0) ? 0.01f : -0.01f) * Const::sol * bf;
    }
  }

  void InitialStatePlex::init_momentum_vec_and_k(const MPF& phi, const MPI& chg, const MPF& bf) {
    init_momentum_vec_and_k(phi, chg, 1.0f);
    for (int i = 0; i < inv_k.kTotSize; ++i)
      inv_k[i] *= bf[i];
  }

  // propagate to radius; returns number of failed propagations
  int InitialStatePlex::propagate_to_r(
      PropAlgo_e algo, const MPF& R, StatePlex& c, bool update_momentum, int N_proc) const {
    // Effective target radius, clamped to what the trajectory can actually
    // reach, and the per-lane flag saying it had to be clamped.
    MPF R_eff = R;
    MPI unreachable(0);

    switch (algo) {
      case PA_Line: {
      }
      case PA_Quadratic: {
      }

      case PA_Exact: {
        const MPF k = 1.0f / inv_k;

        c = *this;

        // Closed-form intersection of the trajectory circle (centre
        // C = (x - k py, y + k px), radius R_c) with the cylinder |P| = R.
        // Subtracting the two circle equations gives the line
        // P.C = (R^2 + d_c^2 - R_c^2) / 2, so with Chat = C / d_c
        //     P = a Chat +/- b Chat_perp,
        //     a = (R^2 + d_c^2 - R_c^2) / (2 d_c),   b = sqrt(R^2 - a^2).
        // d_c^2 - R_c^2 is computed as r0^2 - 2 k (x py - y px): for a stiff
        // track d_c ~ R_c ~ 87.8 pT cm, and the direct difference of their
        // squares loses all float precision.
        const MPF cx = x - k * py;
        const MPF cy = y + k * px;
        const MPF d_c = Matriplex::hypot(cx, cy);
        const MPF R_c = Matriplex::abs(k) * Matriplex::hypot(px, py);
        const MPF dc2_m_Rc2 = x * x + y * y - 2.0f * k * (x * py - y * px);

        // Reachability, see kReachMargin: |a| <= R is the same condition as
        // |d_c - R_c| <= R <= d_c + R_c. An unreachable R is clamped to the band
        // edge, which gives the point of closest approach on the trajectory.
        for (int i = 0; i < N_proc; ++i) {
          float r_lo = std::abs(d_c[i] - R_c[i]) + kReachMargin;
          float r_hi = d_c[i] + R_c[i] - kReachMargin;
          if (r_lo > r_hi) {
            // Band thinner than twice the margin -- a nearly closed loop that
            // barely spans any radius. Aim at its middle.
            r_lo = r_hi = 0.5f * (std::abs(d_c[i] - R_c[i]) + d_c[i] + R_c[i]);
          }
          if (R[i] < r_lo) {
            R_eff[i] = r_lo;
            unreachable[i] = 1;
          } else if (R[i] > r_hi) {
            R_eff[i] = r_hi;
            unreachable[i] = 1;
          }
        }

        const MPF R2 = R_eff * R_eff;
        const MPF a = (R2 + dc2_m_Rc2) / (2.0f * d_c);
        const MPF b = Matriplex::sqrt(Matriplex::max(R2 - a * a, MPF(0.0f)));

        const MPF inv_dc = 1.0f / d_c;
        const MPF hx = cx * inv_dc, hy = cy * inv_dc;  // Chat; Chat_perp = (-hy, hx)

        // Radius vector of the current point and of the two candidate crossings.
        // cos and sin of the turning angle come straight out of the dot and
        // cross products, so no trig is needed for the position or the
        // momentum -- only for alpha itself, which z and dalpha need.
        const MPF u0x = x - cx, u0y = y - cy;
        const MPF inv_Rc2 = 1.0f / (R_c * R_c);

        const MPF p1x = a * hx - b * hy, p1y = a * hy + b * hx;
        const MPF p2x = a * hx + b * hy, p2y = a * hy - b * hx;
        const MPF u1x = p1x - cx, u1y = p1y - cy;
        const MPF u2x = p2x - cx, u2y = p2y - cy;
        const MPF cos1 = (u0x * u1x + u0y * u1y) * inv_Rc2;
        const MPF sin1 = (u0x * u1y - u0y * u1x) * inv_Rc2;
        const MPF cos2 = (u0x * u2x + u0y * u2y) * inv_Rc2;
        const MPF sin2 = (u0x * u2y - u0y * u2x) * inv_Rc2;
        const MPF alpha1 = Matriplex::fast_atan2(sin1, cos1);
        const MPF alpha2 = Matriplex::fast_atan2(sin2, cos2);

        // Of the two crossings take the nearer one. atan2 returns in (-pi, pi],
        // so this is the smaller turn in either direction -- the standard
        // choice, and wrong only for a track already beyond salvage.
        MPF alpha, sina, cosa;
        for (int i = 0; i < N_proc; ++i) {
          const bool first = std::abs(alpha1[i]) <= std::abs(alpha2[i]);
          alpha[i] = first ? alpha1[i] : alpha2[i];
          sina[i] = first ? sin1[i] : sin2[i];
          cosa[i] = first ? cos1[i] : cos2[i];
          c.x[i] = first ? p1x[i] : p2x[i];
          c.y[i] = first ? p1y[i] : p2y[i];
        }

        c.z = z + k * pz * alpha;
        c.dalpha = dalpha + alpha;

        if (update_momentum) {
          // The momentum rotates by exactly the same alpha as the radius vector.
          const MPF o_px = px;
          c.px = px * cosa - py * sina;
          c.py = py * cosa + o_px * sina;
          c.pz = pz;
        }
      }
    }

    // fail_flag: 2 = target radius not on the trajectory at all (pre-checked,
    // c was clamped to the closest reachable radius), 1 = solved but landed
    // further than kReachTolerance from the target, which only round-off should
    // cause.
    MPF r = Matriplex::hypot(c.x, c.y);
    c.fail_flag = 0;
    int n_fail = 0;
    for (int i = 0; i < N_proc; ++i) {
      if (unreachable[i]) {
        c.fail_flag[i] = 2;
        ++n_fail;
      } else if (std::abs(R_eff[i] - r[i]) > kReachTolerance) {
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
        Matriplex::fast_sincos(alpha, sina, cosa);

        c.dalpha = dalpha + alpha;
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

  int InitialStatePlex::propagate_to_plane(
      PropAlgo_e algo, const MP3V& pos, const MP3V& zdir, StatePlex& c, bool update_momentum) const {
    switch (algo) {
      case PA_Line: {
        // Straight-line step along the momentum onto the plane; the momentum
        // is not changed.
        // t * p_vec intersects the plane:
        MPF t = ((pos(0, 0) - x) * zdir(0, 0) + (pos(1, 0) - y) * zdir(1, 0) + (pos(2, 0) - z) * zdir(2, 0)) /
                (px * zdir(0, 0) + py * zdir(1, 0) + pz * zdir(2, 0));

        c = *this;
        c.x += t * c.px;
        c.y += t * c.py;
        c.z += t * c.pz;

        // Signed helix turning angle over the same transverse path, t pT / (k pT).
        c.dalpha += t * inv_k;
        break;
      }

      case PA_Quadratic: {
        throw std::runtime_error("Quadratic prop_to_plane not implemented");
      }

      case PA_Exact: {
        throw std::runtime_error("Exact prop_to_plane not implemented");
      }
    }
    c.fail_flag = 0;
    return 0;
  }

  //===========================================================================
  // Hermite3D interpolation
  //===========================================================================

  void Hermite3D::hermite_1d(const MPF& x1, const MPF& p1, const MPF& x2, const MPF& p2, const MPF& derfac, MPlex4V& h) {
    MPF P = x2 - x1;
    MPF Q = p1 * derfac;
    MPF R = p2 * derfac - 2.0f * P + Q;
    h.aij(0, 0) = x1;
    h.aij(1, 0) = Q;
    h.aij(2, 0) = P - Q - R;
    h.aij(3, 0) = R;
  }

  void Hermite3D::calculate_coeffs(const StatePlex& sp1, const StatePlex& sp2, const MPF& inv_k) {
    m_Hderfac = (sp2.dalpha - sp1.dalpha) / inv_k;
    hermite_1d(sp1.x, sp1.px, sp2.x, sp2.px, m_Hderfac, m_Hx);
    hermite_1d(sp1.y, sp1.py, sp2.y, sp2.py, m_Hderfac, m_Hy);
    hermite_1d(sp1.z, sp1.pz, sp2.z, sp2.pz, m_Hderfac, m_Hz);
    m_Hderfac = 1.0f / m_Hderfac;
  }

  void Hermite3D::calculate_coeffs(const StatePlex& sp, const MPF& inv_k, const MPF& dalpha) {
    // One-point / extrapolating mode -- see the comment on the declaration.
    //
    // Substituting a = dalpha * t into the Taylor series of the exact helix
    // about a = 0, with K = k * dalpha:
    //   r(t) = r0 + K p t + (K dalpha / 2) (-py, px, 0) t^2
    //                     - (K dalpha^2 / 6) ( px, py, 0) t^3
    // z is linear in a, so its quadratic and cubic terms vanish identically.
    const MPF K = dalpha / inv_k;                          // = k * dalpha
    const MPF Kp2 = 0.5f * K * dalpha;                     // +K dalpha / 2
    const MPF Kn2 = -0.5f * K * dalpha;                    // -K dalpha / 2
    const MPF Kn3 = -(1.0f / 6.0f) * K * dalpha * dalpha;  // -K dalpha^2 / 6
    const MPF zero = 0.0f;

    m_Hx.aij(0, 0) = sp.x;
    m_Hx.aij(1, 0) = K * sp.px;
    m_Hx.aij(2, 0) = Kn2 * sp.py;
    m_Hx.aij(3, 0) = Kn3 * sp.px;

    m_Hy.aij(0, 0) = sp.y;
    m_Hy.aij(1, 0) = K * sp.py;
    m_Hy.aij(2, 0) = Kp2 * sp.px;
    m_Hy.aij(3, 0) = Kn3 * sp.py;

    m_Hz.aij(0, 0) = sp.z;
    m_Hz.aij(1, 0) = K * sp.pz;
    m_Hz.aij(2, 0) = zero;
    m_Hz.aij(3, 0) = zero;

    m_Hderfac = 1.0f / K;
  }

  void Hermite3D::evaluate(const MPF& t, MPF& x, MPF& y, MPF& z) const {
    MPF t2 = t * t;
    MPF t3 = t2 * t;
    x = m_Hx(0, 0) + t * m_Hx(1, 0) + t2 * m_Hx(2, 0) + t3 * m_Hx(3, 0);
    y = m_Hy(0, 0) + t * m_Hy(1, 0) + t2 * m_Hy(2, 0) + t3 * m_Hy(3, 0);
    z = m_Hz(0, 0) + t * m_Hz(1, 0) + t2 * m_Hz(2, 0) + t3 * m_Hz(3, 0);
  }

  void Hermite3D::evaluate(const MPF& t, MPF& x, MPF& y, MPF& z, MPF& dx, MPF& dy, MPF& dz) const {
    MPF t2 = t * t;
    MPF t3 = t2 * t;
    x = m_Hx(0, 0) + t * m_Hx(1, 0) + t2 * m_Hx(2, 0) + t3 * m_Hx(3, 0);
    y = m_Hy(0, 0) + t * m_Hy(1, 0) + t2 * m_Hy(2, 0) + t3 * m_Hy(3, 0);
    z = m_Hz(0, 0) + t * m_Hz(1, 0) + t2 * m_Hz(2, 0) + t3 * m_Hz(3, 0);
    dx = (m_Hx(1, 0) + 2.0f * t * m_Hx(2, 0) + 3.0f * t2 * m_Hx(3, 0)) * m_Hderfac;
    dy = (m_Hy(1, 0) + 2.0f * t * m_Hy(2, 0) + 3.0f * t2 * m_Hy(3, 0)) * m_Hderfac;
    dz = (m_Hz(1, 0) + 2.0f * t * m_Hz(2, 0) + 3.0f * t2 * m_Hz(3, 0)) * m_Hderfac;
  }

  void Hermite3D::evaluate(const MPF& t, StatePlex& sp) const { evaluate(t, sp.x, sp.y, sp.z, sp.px, sp.py, sp.pz); }

  //===========================================================================
  // Hermite3DOnPlane intersection
  //===========================================================================

  void Hermite3DOnPlane::init_coeffs(const Hermite3D& h, const MP3V& pos, const MP3V& zdir) {
    m_D = 0.0f;
    m_D.aij(0, 0) = (pos(0, 0) - h.m_Hx(0, 0)) * zdir(0, 0) + (pos(1, 0) - h.m_Hy(0, 0)) * zdir(1, 0) +
                    (pos(2, 0) - h.m_Hz(0, 0)) * zdir(2, 0);
    for (int i = 1; i < 4; ++i) {
      m_D.aij(i, 0) = (h.m_Hx(i, 0) * zdir(0, 0) + h.m_Hy(i, 0) * zdir(1, 0) + h.m_Hz(i, 0) * zdir(2, 0)).negate();
    }

    MPF d0, d1;
    evaluate(0.0f, d0);
    evaluate(1.0f, d1);
    m_T = d0 / (d0 - d1);
  }

  void Hermite3DOnPlane::evaluate(const MPF& t, MPF& d) const {
    MPF t2 = t * t;
    d = m_D(0, 0) + m_D(1, 0) * t + m_D(2, 0) * t2 + m_D(3, 0) * t2 * t;
  }

  void Hermite3DOnPlane::evaluate(const MPF& t, MPF& d, MPF& dddt) const {
    MPF t2 = t * t;
    d = m_D(0, 0) + m_D(1, 0) * t + m_D(2, 0) * t2 + m_D(3, 0) * t2 * t;
    dddt = m_D(1, 0) + 2.0f * m_D(2, 0) * t + 3.0f * m_D(3, 0) * t2;
  }

  void Hermite3DOnPlane::solve() {
    MPF f, df;
    evaluate(m_T, f, df);
    m_T -= f / df;
  }

}  // namespace mkfit::mini_propagators
