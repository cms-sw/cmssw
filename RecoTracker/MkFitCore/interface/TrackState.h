#ifndef RecoTracker_MkFitCore_interface_TrackState_h
#define RecoTracker_MkFitCore_interface_TrackState_h

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/MatrixSTypes.h"
#include "RecoTracker/MkFitCore/interface/MathInlineFunctions.h"

#include <cmath>

namespace mkfit {

  //==============================================================================
  // TrackState
  //==============================================================================

  struct TrackState {
  public:
    TrackState() : charge(0), valid(true) {}
    TrackState(int charge, const SVector3& pos, const SVector3& mom, const SMatrixSym66& err)
        : parameters(SVector6(pos.At(0), pos.At(1), pos.At(2), mom.At(0), mom.At(1), mom.At(2))),
          errors(err),
          charge(charge),
          valid(true) {}
    SVector3 position() const { return SVector3(parameters[0], parameters[1], parameters[2]); }
    SVector6 parameters;
    SMatrixSym66 errors;
    short charge;
    bool valid;

    // Access for packing into Matriplexes
    const float* parArray() const { return parameters.Array(); }
    const float* errArray() const { return errors.Array(); }
    // And non-const for copying back out.
    float* parArray_nc() { return parameters.Array(); }
    float* errArray_nc() { return errors.Array(); }

    // track state position
    float x() const { return parameters.At(0); }
    float y() const { return parameters.At(1); }
    float z() const { return parameters.At(2); }
    float posR() const { return getHypot(x(), y()); }
    float posRsq() const { return x() * x() + y() * y(); }
    float posPhi() const { return getPhi(x(), y()); }
    float posEta() const { return getEta(posR(), z()); }

    // track state position errors
    float exx() const { return std::sqrt(errors.At(0, 0)); }
    float eyy() const { return std::sqrt(errors.At(1, 1)); }
    float ezz() const { return std::sqrt(errors.At(2, 2)); }
    float exy() const { return std::sqrt(errors.At(0, 1)); }
    float exz() const { return std::sqrt(errors.At(0, 2)); }
    float eyz() const { return std::sqrt(errors.At(1, 2)); }

    float eposR() const { return std::sqrt(getRadErr2(x(), y(), errors.At(0, 0), errors.At(1, 1), errors.At(0, 1))); }
    float eposPhi() const { return std::sqrt(getPhiErr2(x(), y(), errors.At(0, 0), errors.At(1, 1), errors.At(0, 1))); }
    float eposEta() const {
      return std::sqrt(getEtaErr2(x(),
                                  y(),
                                  z(),
                                  errors.At(0, 0),
                                  errors.At(1, 1),
                                  errors.At(2, 2),
                                  errors.At(0, 1),
                                  errors.At(0, 2),
                                  errors.At(1, 2)));
    }

    // track state momentum
    float invpT() const { return parameters.At(3); }
    float momPhi() const { return parameters.At(4); }
    float theta() const { return parameters.At(5); }
    float pT() const { return std::abs(1.f / parameters.At(3)); }
    float px() const { return pT() * std::cos(parameters.At(4)); }
    float py() const { return pT() * std::sin(parameters.At(4)); }
    float pz() const { return pT() / std::tan(parameters.At(5)); }
    float momEta() const { return getEta(theta()); }
    float p() const { return pT() / std::sin(parameters.At(5)); }

    float einvpT() const { return std::sqrt(errors.At(3, 3)); }
    float emomPhi() const { return std::sqrt(errors.At(4, 4)); }
    float etheta() const { return std::sqrt(errors.At(5, 5)); }
    float epT() const { return std::sqrt(errors.At(3, 3)) / (parameters.At(3) * parameters.At(3)); }
    float emomEta() const { return std::sqrt(errors.At(5, 5)) / std::sin(parameters.At(5)); }
    float epxpx() const { return std::sqrt(getPxPxErr2(invpT(), momPhi(), errors.At(3, 3), errors.At(4, 4))); }
    float epypy() const { return std::sqrt(getPyPyErr2(invpT(), momPhi(), errors.At(3, 3), errors.At(4, 4))); }
    float epzpz() const { return std::sqrt(getPzPzErr2(invpT(), theta(), errors.At(3, 3), errors.At(5, 5))); }

    void convertFromCartesianToCCS();
    void convertFromCCSToCartesian();
    SMatrix66 jacobianCCSToCartesian(float invpt, float phi, float theta) const;
    SMatrix66 jacobianCartesianToCCS(float px, float py, float pz) const;

    void convertFromGlbCurvilinearToCCS();
    void convertFromCCSToGlbCurvilinear();
    //last row/column are zeros
    SMatrix66 jacobianCCSToCurvilinear(float invpt, float cosP, float sinP, float cosT, float sinT, short charge) const;
    SMatrix66 jacobianCurvilinearToCCS(float px, float py, float pz, short charge) const;

    bool hasNanNSillyValues() const;
  };

}  // namespace mkfit

#endif  // RecoTracker_MkFitCore_interface_TrackState_h
