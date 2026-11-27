#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"
#include "RecoTracker/MkFitCore/interface/TrackState.h"
#include "Matrix.h"

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
    //assume we are currently in ccs coordinates and want to move to global state with curvilinear error
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

  bool TrackState::hasNanNSillyValues() const {
    bool is_silly = false;
    for (int i = 0; i < LL; ++i) {
      for (int j = 0; j <= i; ++j) {
        if ((i == j && errors.At(i, j) < 0) || !isFinite(errors.At(i, j))) {
          is_silly = true;
          return is_silly;
        }
      }
    }
    return is_silly;
  }

}  // namespace mkfit
