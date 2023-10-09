#define G__DICTIONARY

#define SMATRIX_USE_CONSTEXPR
#include <Math/SVector.h>
#include <Math/SMatrix.h>

#include "DataFormats/Math/interface/LorentzVector.h"
#include "Rtypes.h"
#include "Math/PxPyPzE4D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Math/Cartesian3D.h"
#include "Math/Polar3D.h"
#include "Math/CylindricalEta3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Matrix.h"
#include "DataFormats/Math/interface/libminifloat.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include <vector>

namespace DataFormats_Math {
  struct dictionary {
    math::XYZVector v1;
    math::XYZVectorD vd1;
    math::XYZVectorF vf1;
    math::RhoEtaPhiVector v2;
    math::RhoEtaPhiVectorD vd2;
    math::RhoEtaPhiVectorF vf2;
    //    math::RThetaPhiVector v3;
    //    math::RThetaPhiVectorD vd3;
    //    math::RThetaPhiVectorF vf3;
    math::XYZPoint p1;
    math::XYZPointD pd1;
    math::XYZPointF pf1;
    math::PtEtaPhiELorentzVector l1;
    math::PtEtaPhiELorentzVectorD ld1;
    math::PtEtaPhiELorentzVectorF lf1;
    math::PtEtaPhiMLorentzVector l3;
    math::PtEtaPhiMLorentzVectorD ld3;
    math::PtEtaPhiMLorentzVectorF lf3;
    math::XYZTLorentzVector l2;
    math::XYZTLorentzVectorD ld2;
    math::XYZTLorentzVectorF lf2;

    math::LocalPoint lp1;
    math::GlobalPoint gp1;
    math::LocalVector lv1;
    math::GlobalVector gv1;

    std::vector<math::XYZVector> vv1;
    std::vector<math::XYZVectorD> vvd1;
    std::vector<math::XYZVectorF> vvf1;
    std::vector<math::RhoEtaPhiVector> vv2;
    std::vector<math::RhoEtaPhiVectorD> vvd2;
    std::vector<math::RhoEtaPhiVectorF> vvf2;
    //     std::vector<math::RThetaPhiVector> vv3;
    //     std::vector<math::RThetaPhiVectorD> vvd3;
    //     std::vector<math::RThetaPhiVectorF> vvf3;
    std::vector<math::XYZPoint> vp1;
    std::vector<math::XYZPointD> vpd1;
    std::vector<math::XYZPointF> vpf1;
    std::vector<math::PtEtaPhiELorentzVector> vl1;
    std::vector<math::PtEtaPhiELorentzVectorD> vld1;
    std::vector<math::PtEtaPhiELorentzVectorF> vlf1;
    std::vector<math::PtEtaPhiMLorentzVector> vl3;
    std::vector<math::PtEtaPhiMLorentzVectorD> vld3;
    std::vector<math::PtEtaPhiMLorentzVectorF> vlf3;
    std::vector<math::XYZTLorentzVector> vl2;
    std::vector<math::XYZTLorentzVectorD> vld2;
    std::vector<math::XYZTLorentzVectorF> vlf2;

    edm::Wrapper<math::XYZVector> wv1;
    edm::Wrapper<math::XYZVectorD> wvd1;
    edm::Wrapper<math::XYZVectorF> wvf1;
    edm::Wrapper<math::RhoEtaPhiVector> wv2;
    edm::Wrapper<math::RhoEtaPhiVectorD> wvd2;
    edm::Wrapper<math::RhoEtaPhiVectorF> wvf2;
    //     edm::Wrapper<math::RThetaPhiVector> wv3;
    //     edm::Wrapper<math::RThetaPhiVectorD> wvd3;
    //     edm::Wrapper<math::RThetaPhiVectorF> wvf3;
    edm::Wrapper<math::XYZPoint> wp1;
    edm::Wrapper<math::XYZPointD> wpd1;
    edm::Wrapper<math::XYZPointF> wpf1;
    edm::Wrapper<math::PtEtaPhiELorentzVector> wl1;
    edm::Wrapper<math::PtEtaPhiELorentzVectorD> wld1;
    edm::Wrapper<math::PtEtaPhiELorentzVectorF> wlf1;
    edm::Wrapper<math::PtEtaPhiMLorentzVector> wl3;
    edm::Wrapper<math::PtEtaPhiMLorentzVectorD> wld3;
    edm::Wrapper<math::PtEtaPhiMLorentzVectorF> wlf3;
    edm::Wrapper<math::XYZTLorentzVector> wl2;
    edm::Wrapper<math::XYZTLorentzVectorD> wld2;
    edm::Wrapper<math::XYZTLorentzVectorF> wlf2;

    edm::Wrapper<std::vector<math::XYZVector> > wvvv1;
    edm::Wrapper<std::vector<math::XYZVectorD> > wvvd1;
    edm::Wrapper<std::vector<math::XYZVectorF> > wvvf1;
    edm::Wrapper<std::vector<math::RhoEtaPhiVector> > wvv2;
    edm::Wrapper<std::vector<math::RhoEtaPhiVectorD> > wvvd2;
    edm::Wrapper<std::vector<math::RhoEtaPhiVectorF> > wvvf2;
    //  edm::Wrapper<std::vector<math::RThetaPhiVector> > wvv3;
    //     edm::Wrapper<std::vector<math::RThetaPhiVectorD> > wvvd3;
    //     edm::Wrapper<std::vector<math::RThetaPhiVectorF> > wvvf3;
    edm::Wrapper<std::vector<math::XYZPoint> > wvp1;
    edm::Wrapper<std::vector<math::XYZPointD> > wvpd1;
    edm::Wrapper<std::vector<math::XYZPointF> > wvpf1;
    edm::Wrapper<std::vector<math::PtEtaPhiELorentzVector> > wvl1;
    edm::Wrapper<std::vector<math::PtEtaPhiELorentzVectorD> > wvld1;
    edm::Wrapper<std::vector<math::PtEtaPhiELorentzVectorF> > wvlf1;
    edm::Wrapper<std::vector<math::PtEtaPhiMLorentzVector> > wvl3;
    edm::Wrapper<std::vector<math::PtEtaPhiMLorentzVectorD> > wvld3;
    edm::Wrapper<std::vector<math::PtEtaPhiMLorentzVectorF> > wvlf3;
    edm::Wrapper<std::vector<math::XYZTLorentzVector> > wvl2;
    edm::Wrapper<std::vector<math::XYZTLorentzVectorD> > wvld2;
    edm::Wrapper<std::vector<math::XYZTLorentzVectorF> > wvlf2;

    //    math::Vector<1>::type vV1;
    math::Vector<2>::type vV2;
    math::Vector<3>::type vV3;
    math::Vector<4>::type vV4;
    math::Vector<5>::type vV5;
    math::Vector<6>::type vV6;
    //    math::VectorD<1>::type vVd1;
    math::VectorD<2>::type vVd2;
    math::VectorD<3>::type vVd3;
    math::VectorD<4>::type vVd4;
    math::VectorD<5>::type vVd5;
    math::VectorD<6>::type vVd6;
    //    math::VectorF<1>::type vVf1;
    math::VectorF<2>::type vVf2;
    math::VectorF<3>::type vVf3;
    math::VectorF<4>::type vVf4;
    math::VectorF<5>::type vVf5;
    math::VectorF<6>::type vVf6;
    math::VectorF<7>::type vVf7;

    //    math::Error<1>::type e1;
    math::Error<2>::type e2;
    math::Error<3>::type e3;
    math::Error<4>::type e4;
    math::Error<5>::type e5;
    math::Error<6>::type e6;
    math::Error<7>::type e7;

    //     math::ErrorD<1>::type ed1;
    math::ErrorD<2>::type ed2;
    math::ErrorD<3>::type ed3;
    math::ErrorD<4>::type ed4;
    math::ErrorD<5>::type ed5;
    math::ErrorD<6>::type ed6;
    //    math::ErrorF<1>::type ef1;
    math::ErrorF<2>::type ef2;
    math::ErrorF<3>::type ef3;
    math::ErrorF<4>::type ef4;
    math::ErrorF<5>::type ef5;
    math::ErrorF<6>::type ef6;
    math::ErrorF<7>::type ef7;

    //     std::vector<math::Error<1>::type> ve1;
    std::vector<math::Error<2>::type> ve2;
    std::vector<math::Error<3>::type> ve3;
    std::vector<math::Error<4>::type> ve4;
    std::vector<math::Error<5>::type> ve5;
    std::vector<math::Error<6>::type> ve6;
    std::vector<math::Error<7>::type> ve7;

    //     edm::Wrapper<math::Error<1>::type> we1;
    edm::Wrapper<math::Error<2>::type> we2;
    edm::Wrapper<math::Error<3>::type> we3;
    edm::Wrapper<math::Error<4>::type> we4;
    edm::Wrapper<math::Error<5>::type> we5;
    edm::Wrapper<math::Error<6>::type> we6;
    edm::Wrapper<math::Error<7>::type> we7;

    //     edm::Wrapper<std::vector<math::Error<1>::type> > wve1;
    edm::Wrapper<std::vector<math::Error<2>::type> > wve2;
    edm::Wrapper<std::vector<math::Error<3>::type> > wve3;
    edm::Wrapper<std::vector<math::Error<4>::type> > wve4;
    edm::Wrapper<std::vector<math::Error<5>::type> > wve5;
    edm::Wrapper<std::vector<math::Error<6>::type> > wve6;
    edm::Wrapper<std::vector<math::Error<7>::type> > wve7;

    edm::RefVector<std::vector<math::PtEtaPhiELorentzVector> > rvl2;
    edm::RefVector<std::vector<math::PtEtaPhiELorentzVectorD> > rvld2;
    edm::RefVector<std::vector<math::PtEtaPhiELorentzVectorF> > rvlf2;

    //     ROOT::Math::MatRepSym<Double32_t, 1> sm1;
    ROOT::Math::MatRepSym<Double32_t, 2> sm2;
    ROOT::Math::MatRepSym<Double32_t, 3> sm3;
    ROOT::Math::MatRepSym<Double32_t, 4> sm4;
    ROOT::Math::MatRepSym<Double32_t, 5> sm5;
    ROOT::Math::MatRepSym<Double32_t, 6> sm6;
    ROOT::Math::MatRepSym<Double32_t, 7> sm7;
    //    ROOT::Math::MatRepSym<double, 1> smd1;
    ROOT::Math::MatRepSym<double, 2> smd2;
    ROOT::Math::MatRepSym<double, 3> smd3;
    ROOT::Math::MatRepSym<double, 4> smd4;
    ROOT::Math::MatRepSym<double, 5> smd5;
    ROOT::Math::MatRepSym<double, 6> smd6;
    ROOT::Math::MatRepSym<double, 7> smd7;
    //    ROOT::Math::MatRepSym<float, 1> smf1;
    ROOT::Math::MatRepSym<float, 2> smf2;
    ROOT::Math::MatRepSym<float, 3> smf3;
    ROOT::Math::MatRepSym<float, 4> smf4;
    ROOT::Math::MatRepSym<float, 5> smf5;
    ROOT::Math::MatRepSym<float, 6> smf6;
    ROOT::Math::MatRepSym<float, 7> smf7;

    //Used by ECAL Weights reconstruction
    //Used by ECAL Weights reconstruction
    math::Matrix<3, 10>::type mw;
    math::Matrix<10, 10>::type mcw;
    math::Matrix<2, 3>::type mw1;
    ROOT::Math::MatRepStd<double, 3, 10> smdw;
    ROOT::Math::MatRepStd<double, 10, 10> smdcw;
    ROOT::Math::MatRepStd<double, 2, 3> smdcw1;

    //Used by TauReco
    //    std::pair<ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag>,float> calotti_ppf;
    std::vector<
        std::pair<ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>, ROOT::Math::DefaultCoordinateSystemTag>,
                  float> >
        vcalotti_ppf;

    // Used by CommonTools/PileupModules
    edm::ValueMap<math::XYZTLorentzVector> vmp4;
    edm::Wrapper<edm::ValueMap<math::XYZTLorentzVector> > wvmp4;
  };
  MiniFloatConverter::ReduceMantissaToNbitsRounding red(12);
}  // namespace DataFormats_Math
