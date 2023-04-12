#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "Matrix.h"

//#define DEBUG
#include "Debug.h"

namespace mkfit {

  //==============================================================================
  // TrackState
  //==============================================================================

  void TrackState::convertFromCartesianToCCS() {
    //assume we are currently in cartesian coordinates and want to move to ccs
    const float px = parameters.At(3);
    const float py = parameters.At(4);
    const float pz = parameters.At(5);
    const float pt = std::sqrt(px * px + py * py);
    const float phi = getPhi(px, py);
    const float theta = getTheta(pt, pz);
    parameters.At(3) = 1.f / pt;
    parameters.At(4) = phi;
    parameters.At(5) = theta;
    SMatrix66 jac = jacobianCartesianToCCS(px, py, pz);
    errors = ROOT::Math::Similarity(jac, errors);
  }

  void TrackState::convertFromCCSToCartesian() {
    //assume we are currently in ccs coordinates and want to move to cartesian
    const float invpt = parameters.At(3);
    const float phi = parameters.At(4);
    const float theta = parameters.At(5);
    const float pt = 1.f / invpt;
    float cosP = std::cos(phi);
    float sinP = std::sin(phi);
    float cosT = std::cos(theta);
    float sinT = std::sin(theta);
    parameters.At(3) = cosP * pt;
    parameters.At(4) = sinP * pt;
    parameters.At(5) = cosT * pt / sinT;
    SMatrix66 jac = jacobianCCSToCartesian(invpt, phi, theta);
    errors = ROOT::Math::Similarity(jac, errors);
  }

  SMatrix66 TrackState::jacobianCCSToCartesian(float invpt, float phi, float theta) const {
    //arguments are passed so that the function can be used both starting from ccs and from cartesian
    SMatrix66 jac = ROOT::Math::SMatrixIdentity();
    float cosP = std::cos(phi);
    float sinP = std::sin(phi);
    float cosT = std::cos(theta);
    float sinT = std::sin(theta);
    const float pt = 1.f / invpt;
    jac(3, 3) = -cosP * pt * pt;
    jac(3, 4) = -sinP * pt;
    jac(4, 3) = -sinP * pt * pt;
    jac(4, 4) = cosP * pt;
    jac(5, 3) = -cosT * pt * pt / sinT;
    jac(5, 5) = -pt / (sinT * sinT);
    return jac;
  }

  SMatrix66 TrackState::jacobianCartesianToCCS(float px, float py, float pz) const {
    //arguments are passed so that the function can be used both starting from ccs and from cartesian
    SMatrix66 jac = ROOT::Math::SMatrixIdentity();
    const float pt = std::sqrt(px * px + py * py);
    const float p2 = px * px + py * py + pz * pz;
    jac(3, 3) = -px / (pt * pt * pt);
    jac(3, 4) = -py / (pt * pt * pt);
    jac(4, 3) = -py / (pt * pt);
    jac(4, 4) = px / (pt * pt);
    jac(5, 3) = px * pz / (pt * p2);
    jac(5, 4) = py * pz / (pt * p2);
    jac(5, 5) = -pt / p2;
    return jac;
  }

  void TrackState::convertFromGlbCurvilinearToCCS() {
    //assume we are currently in global state with curvilinear error and want to move to ccs
    const float px = parameters.At(3);
    const float py = parameters.At(4);
    const float pz = parameters.At(5);
    const float pt = std::sqrt(px * px + py * py);
    const float phi = getPhi(px, py);
    const float theta = getTheta(pt, pz);
    parameters.At(3) = 1.f / pt;
    parameters.At(4) = phi;
    parameters.At(5) = theta;
    SMatrix66 jac = jacobianCurvilinearToCCS(px, py, pz, charge);
    errors = ROOT::Math::Similarity(jac, errors);
  }

  void TrackState::convertFromCCSToGlbCurvilinear() {
    //assume we are currently in ccs coordinates and want to move to global state with cartesian error
    const float invpt = parameters.At(3);
    const float phi = parameters.At(4);
    const float theta = parameters.At(5);
    const float pt = 1.f / invpt;
    float cosP = std::cos(phi);
    float sinP = std::sin(phi);
    float cosT = std::cos(theta);
    float sinT = std::sin(theta);
    parameters.At(3) = cosP * pt;
    parameters.At(4) = sinP * pt;
    parameters.At(5) = cosT * pt / sinT;
    SMatrix66 jac = jacobianCCSToCurvilinear(invpt, cosP, sinP, cosT, sinT, charge);
    errors = ROOT::Math::Similarity(jac, errors);
  }

  SMatrix66 TrackState::jacobianCCSToCurvilinear(
      float invpt, float cosP, float sinP, float cosT, float sinT, short charge) const {
    SMatrix66 jac;
    jac(3, 0) = -sinP;
    jac(4, 0) = -cosP * cosT;
    jac(3, 1) = cosP;
    jac(4, 1) = -sinP * cosT;
    jac(4, 2) = sinT;
    jac(0, 3) = charge * sinT;
    jac(0, 5) = charge * cosT * invpt;
    jac(1, 5) = -1.f;
    jac(2, 4) = 1.f;

    return jac;
  }

  SMatrix66 TrackState::jacobianCurvilinearToCCS(float px, float py, float pz, short charge) const {
    const float pt2 = px * px + py * py;
    const float pt = sqrt(pt2);
    const float invpt2 = 1.f / pt2;
    const float invpt = 1.f / pt;
    const float invp = 1.f / sqrt(pt2 + pz * pz);
    const float sinPhi = py * invpt;
    const float cosPhi = px * invpt;
    const float sinLam = pz * invp;
    const float cosLam = pt * invp;

    SMatrix66 jac;
    jac(0, 3) = -sinPhi;
    jac(0, 4) = -sinLam * cosPhi;
    jac(1, 3) = cosPhi;
    jac(1, 4) = -sinLam * sinPhi;
    jac(2, 4) = cosLam;
    jac(3, 0) = charge / cosLam;  //assumes |charge|==1 ; else 1.f/charge here
    jac(3, 1) = pz * invpt2;
    jac(4, 2) = 1.f;
    jac(5, 1) = -1.f;

    return jac;
  }

  //==============================================================================
  // TrackBase
  //==============================================================================

  bool TrackBase::hasSillyValues(bool dump, bool fix, const char* pref) {
    bool is_silly = false;
    for (int i = 0; i < LL; ++i) {
      for (int j = 0; j <= i; ++j) {
        if ((i == j && state_.errors.At(i, j) < 0) || !isFinite(state_.errors.At(i, j))) {
          if (!is_silly) {
            is_silly = true;
            if (dump)
              printf("%s (label=%d, pT=%f):", pref, label(), pT());
          }
          if (dump)
            printf(" (%d,%d)=%e", i, j, state_.errors.At(i, j));
          if (fix)
            state_.errors.At(i, j) = 0.00001;
        }
      }
    }
    if (is_silly && dump)
      printf("\n");
    return is_silly;
  }

  bool TrackBase::hasNanNSillyValues() const {
    bool is_silly = false;
    for (int i = 0; i < LL; ++i) {
      for (int j = 0; j <= i; ++j) {
        if ((i == j && state_.errors.At(i, j) < 0) || !isFinite(state_.errors.At(i, j))) {
          is_silly = true;
          return is_silly;
        }
      }
    }
    return is_silly;
  }

  // If linearize=true, use linear estimate of d0: suitable at pT>~10 GeV (--> 10 micron error)
  float TrackBase::d0BeamSpot(const float x_bs, const float y_bs, bool linearize) const {
    if (linearize) {
      return std::abs(std::cos(momPhi()) * (y() - y_bs) - std::sin(momPhi()) * (x() - x_bs));
    } else {
      const float k = ((charge() < 0) ? 100.0f : -100.0f) / (Const::sol * Config::Bfield);
      const float abs_ooc_half = std::abs(k * pT());
      // center of helix in x,y plane
      const float x_center = x() - k * py();
      const float y_center = y() + k * px();
      return std::hypot(x_center - x_bs, y_center - y_bs) - abs_ooc_half;
    }
  }

  const char* TrackBase::algoint_to_cstr(int algo) {
    static const char* const names[] = {"undefAlgorithm",
                                        "ctf",
                                        "duplicateMerge",
                                        "cosmics",
                                        "initialStep",
                                        "lowPtTripletStep",
                                        "pixelPairStep",
                                        "detachedTripletStep",
                                        "mixedTripletStep",
                                        "pixelLessStep",
                                        "tobTecStep",
                                        "jetCoreRegionalStep",
                                        "conversionStep",
                                        "muonSeededStepInOut",
                                        "muonSeededStepOutIn",
                                        "outInEcalSeededConv",
                                        "inOutEcalSeededConv",
                                        "nuclInter",
                                        "standAloneMuon",
                                        "globalMuon",
                                        "cosmicStandAloneMuon",
                                        "cosmicGlobalMuon",
                                        "highPtTripletStep",
                                        "lowPtQuadStep",
                                        "detachedQuadStep",
                                        "reservedForUpgrades1",
                                        "reservedForUpgrades2",
                                        "bTagGhostTracks",
                                        "beamhalo",
                                        "gsf",
                                        "hltPixel",
                                        "hltIter0",
                                        "hltIter1",
                                        "hltIter2",
                                        "hltIter3",
                                        "hltIter4",
                                        "hltIterX",
                                        "hiRegitMuInitialStep",
                                        "hiRegitMuLowPtTripletStep",
                                        "hiRegitMuPixelPairStep",
                                        "hiRegitMuDetachedTripletStep",
                                        "hiRegitMuMixedTripletStep",
                                        "hiRegitMuPixelLessStep",
                                        "hiRegitMuTobTecStep",
                                        "hiRegitMuMuonSeededStepInOut",
                                        "hiRegitMuMuonSeededStepOutIn",
                                        "algoSize"};

    if (algo < 0 || algo >= (int)TrackAlgorithm::algoSize)
      return names[0];
    return names[algo];
  }

  //==============================================================================
  // Track
  //==============================================================================

  void Track::resizeHitsForInput() {
    bzero(&hitsOnTrk_, sizeof(hitsOnTrk_));
    hitsOnTrk_.resize(lastHitIdx_ + 1);
  }

  void Track::sortHitsByLayer() {
    std::stable_sort(&hitsOnTrk_[0], &hitsOnTrk_[lastHitIdx_ + 1], [](const auto& h1, const auto& h2) {
      return h1.layer < h2.layer;
    });
  }

  float Track::swimPhiToR(const float x0, const float y0) const {
    const float dR = getHypot(x() - x0, y() - y0);
    // XXX-ASSUMPTION-ERROR can not always reach R, should see what callers expect.
    // For now return PI to signal apex on the ohter side of the helix.
    const float v = dR / 176.f / pT() * charge();
    const float dPhi = std::abs(v) <= 1.0f ? 2.f * std::asin(v) : Const::PI;
    ;
    return squashPhiGeneral(momPhi() - dPhi);
  }

  bool Track::canReachRadius(float R) const {
    const float k = ((charge() < 0) ? 100.0f : -100.0f) / (Const::sol * Config::Bfield);
    const float ooc = 2.0f * k * pT();
    return std::abs(ooc) > R - std::hypot(x(), y());
  }

  float Track::maxReachRadius() const {
    const float k = ((charge() < 0) ? 100.0f : -100.0f) / (Const::sol * Config::Bfield);
    const float abs_ooc_half = std::abs(k * pT());
    // center of helix in x,y plane
    const float x_center = x() - k * py();
    const float y_center = y() + k * px();
    return std::hypot(x_center, y_center) + abs_ooc_half;
  }

  float Track::zAtR(float R, float* r_reached) const {
    float xc = x();
    float yc = y();
    float pxc = px();
    float pyc = py();

    const float ipt = invpT();
    const float kinv = ((charge() < 0) ? 0.01f : -0.01f) * Const::sol * Config::Bfield;
    const float k = 1.0f / kinv;

    const float c = 0.5f * kinv * ipt;
    const float ooc = 1.0f / c;  // 2 * radius of curvature
    const float lambda = pz() * ipt;

    //printf("Track::zAtR to R=%f: k=%e, ipt=%e, c=%e, ooc=%e  -- can hit = %f (if > 1 can)\n",
    //       R, k, ipt, c, ooc, ooc / (R - std::hypot(xc,yc)));

    float D = 0;

    for (int i = 0; i < Config::Niter; ++i) {
      // compute tangental and ideal distance for the current iteration.
      // 3-rd order asin for symmetric incidence (shortest arc lenght).
      float r0 = std::hypot(xc, yc);
      float td = (R - r0) * c;
      float id = ooc * td * (1.0f + 0.16666666f * td * td);
      // This would be for line approximation:
      // float id = R - r0;
      D += id;

      //printf("%-3d r0=%f R-r0=%f td=%f id=%f id_line=%f delta_id=%g\n",
      //       i, r0, R-r0, td, id, R - r0, id - (R-r0));

      float cosa = std::cos(id * ipt * kinv);
      float sina = std::sin(id * ipt * kinv);

      //update parameters
      xc += k * (pxc * sina - pyc * (1.0f - cosa));
      yc += k * (pyc * sina + pxc * (1.0f - cosa));

      const float pxo = pxc;  //copy before overwriting
      pxc = pxc * cosa - pyc * sina;
      pyc = pyc * cosa + pxo * sina;
    }

    if (r_reached)
      *r_reached = std::hypot(xc, yc);

    return z() + lambda * D;

    // ----------------------------------------------------------------
    // Exact solution from Avery's notes ... loses precision somewhere
    // {
    //   const float a = kinv;
    //   float pT      = S.pT();

    //   float ax2y2  = a*(x*x + y*y);
    //   float T      = std::sqrt(pT*pT - 2.0f*a*(x*py - y*px) + a*ax2y2);
    //   float D0     = (T - pT) / a;
    //   float D      = (-2.0f * (x*py - y*px) + a * (x*x + y*y)) / (T + pT);

    //   float B      = c * std::sqrt((R*R - D*D) / (1.0f + 2.0f*c*D));
    //   float s1     = std::asin(B) / c;
    //   float s2     = (Const::PI - std::asin(B)) / c;

    //   printf("pt %f, invpT %f\n", pT, S.invpT());
    //   printf("lambda %f, a %f, c %f, T %f, D0 %f, D %f, B %f, s1 %f, s2 %f\n",
    //          lambda, a, c, T, D0, D, B, s1, s2);
    //   printf("%f = %f / %f\n", (R*R - D*D) / (1.0f + 2.0f*c*D), (R*R - D*D), (1.0f + 2.0f*c*D));

    //   z1 = S.z() + lambda * s1;
    //   z2 = S.z() + lambda * s2;

    //   printf("z1=%f z2=%f\n", z1, z2);
    // }
    // ----------------------------------------------------------------
  }

  float Track::rAtZ(float Z) const {
    float xc = x();
    float yc = y();
    float pxc = px();
    float pyc = py();

    const float ipt = invpT();
    const float kinv = ((charge() < 0) ? 0.01f : -0.01f) * Const::sol * Config::Bfield;
    const float k = 1.0f / kinv;

    const float dz = Z - z();
    const float alpha = dz * ipt * kinv * std::tan(theta());

    const float cosa = std::cos(alpha);
    const float sina = std::sin(alpha);

    xc += k * (pxc * sina - pyc * (1.0f - cosa));
    yc += k * (pyc * sina + pxc * (1.0f - cosa));

    // const float pxo = pxc;//copy before overwriting
    // pxc = pxc * cosa  -  pyc * sina;
    // pyc = pyc * cosa  +  pxo * sina;

    return std::hypot(xc, yc);
  }

  //==============================================================================

  void print(const TrackState& s) {
    std::cout << " x:  " << s.parameters[0] << " y:  " << s.parameters[1] << " z:  " << s.parameters[2] << std::endl
              << " px: " << s.parameters[3] << " py: " << s.parameters[4] << " pz: " << s.parameters[5] << std::endl
              << "valid: " << s.valid << " errors: " << std::endl;
    dumpMatrix(s.errors);
    std::cout << std::endl;
  }

  void print(std::string pfx, int itrack, const Track& trk, bool print_hits) {
    std::cout << std::endl
              << pfx << ": " << itrack << " hits: " << trk.nFoundHits() << " label: " << trk.label() << " State"
              << std::endl;
    print(trk.state());
    if (print_hits) {
      for (int i = 0; i < trk.nTotalHits(); ++i)
        printf("  %2d: lyr %2d idx %d\n", i, trk.getHitLyr(i), trk.getHitIdx(i));
    }
  }

  void print(std::string pfx, const TrackState& s) {
    std::cout << pfx << std::endl;
    print(s);
  }

}  // end namespace mkfit
