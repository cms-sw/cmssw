#include <cppunit/extensions/HelperMacros.h>
#include <algorithm>
#include <iterator>
#include <iostream>

#include "DataFormats/PatCandidates/interface/ParametrizationHelper.h"
#include "DataFormats/Math/interface/deltaPhi.h"

class testKinParametrizations : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testKinParametrizations);

  CPPUNIT_TEST(testTrivialVec2Par_Cart);
  CPPUNIT_TEST(testTrivialVec2Par_ECart);
  CPPUNIT_TEST(testTrivialVec2Par_Spher);
  CPPUNIT_TEST(testTrivialVec2Par_ESpher);
  CPPUNIT_TEST(testTrivialVec2Par_MomDev);
  CPPUNIT_TEST(testTrivialVec2Par_EMomDev);
  CPPUNIT_TEST(testTrivialVec2Par_MCCart);
  CPPUNIT_TEST(testTrivialVec2Par_MCSpher);
  CPPUNIT_TEST(testTrivialVec2Par_MCPInvSpher);
  CPPUNIT_TEST(testTrivialVec2Par_EtEtaPhi);
  CPPUNIT_TEST(testTrivialVec2Par_EtThetaPhi);
  CPPUNIT_TEST(testTrivialVec2Par_MCMomDev);
  CPPUNIT_TEST(testTrivialVec2Par_EScaledMomDev);

  CPPUNIT_TEST(testVecDiff2Par_Cart);
  CPPUNIT_TEST(testVecDiff2Par_ECart);
  CPPUNIT_TEST(testVecDiff2Par_Spher);
  CPPUNIT_TEST(testVecDiff2Par_ESpher);
  CPPUNIT_TEST(testVecDiff2Par_MomDev);
  CPPUNIT_TEST(testVecDiff2Par_EMomDev);
  CPPUNIT_TEST(testVecDiff2Par_MCCart);
  CPPUNIT_TEST(testVecDiff2Par_MCSpher);
  CPPUNIT_TEST(testVecDiff2Par_MCPInvSpher);
  CPPUNIT_TEST(testVecDiff2Par_EtEtaPhi);
  CPPUNIT_TEST(testVecDiff2Par_EtThetaPhi);
  CPPUNIT_TEST(testVecDiff2Par_MCMomDev);
  CPPUNIT_TEST(testVecDiff2Par_EScaledMomDev);

  CPPUNIT_TEST(testVecVec2Diff_Cart);
  CPPUNIT_TEST(testVecVec2Diff_ECart);
  CPPUNIT_TEST(testVecVec2Diff_Spher);
  CPPUNIT_TEST(testVecVec2Diff_ESpher);
  CPPUNIT_TEST(testVecVec2Diff_MomDev);
  CPPUNIT_TEST(testVecVec2Diff_EMomDev);
  CPPUNIT_TEST(testVecVec2Diff_MCCart);
  CPPUNIT_TEST(testVecVec2Diff_MCSpher);
  CPPUNIT_TEST(testVecVec2Diff_MCPInvSpher);
  CPPUNIT_TEST(testVecVec2Diff_EtEtaPhi);
  CPPUNIT_TEST(testVecVec2Diff_EtThetaPhi);
  CPPUNIT_TEST(testVecVec2Diff_MCMomDev);
  CPPUNIT_TEST(testVecVec2Diff_EScaledMomDev);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}

  void testTrivialVec2Par_Cart();
  void testTrivialVec2Par_ECart();
  void testTrivialVec2Par_Spher();
  void testTrivialVec2Par_ESpher();
  void testTrivialVec2Par_MomDev();
  void testTrivialVec2Par_EMomDev();
  void testTrivialVec2Par_MCCart();
  void testTrivialVec2Par_MCSpher();
  void testTrivialVec2Par_MCPInvSpher();
  void testTrivialVec2Par_EtEtaPhi();
  void testTrivialVec2Par_EtThetaPhi();
  void testTrivialVec2Par_MCMomDev();
  void testTrivialVec2Par_EScaledMomDev();

  void testVecDiff2Par_Cart();
  void testVecDiff2Par_ECart();
  void testVecDiff2Par_Spher();
  void testVecDiff2Par_ESpher();
  void testVecDiff2Par_MomDev();
  void testVecDiff2Par_EMomDev();
  void testVecDiff2Par_MCCart();
  void testVecDiff2Par_MCSpher();
  void testVecDiff2Par_MCPInvSpher();
  void testVecDiff2Par_EtEtaPhi();
  void testVecDiff2Par_EtThetaPhi();
  void testVecDiff2Par_MCMomDev();
  void testVecDiff2Par_EScaledMomDev();

  void testVecVec2Diff_Cart();
  void testVecVec2Diff_ECart();
  void testVecVec2Diff_Spher();
  void testVecVec2Diff_ESpher();
  void testVecVec2Diff_MomDev();
  void testVecVec2Diff_EMomDev();
  void testVecVec2Diff_MCCart();
  void testVecVec2Diff_MCSpher();
  void testVecVec2Diff_MCPInvSpher();
  void testVecVec2Diff_EtEtaPhi();
  void testVecVec2Diff_EtThetaPhi();
  void testVecVec2Diff_MCMomDev();
  void testVecVec2Diff_EScaledMomDev();

  typedef math::XYZTLorentzVector P4C;
  typedef math::PtEtaPhiMLorentzVector P4P;
  typedef AlgebraicVector4 V4;

private:
  typedef pat::CandKinResolution::Parametrization Parametrization;
  bool testTrivialVec2Par(Parametrization p, int tries = 100000);
  bool testVecDiff2Par(Parametrization p, int tries = 100000);
  bool testVecVec2Diff(Parametrization p, int tries = 100000);

  // Utilities
  static double r(double val = 1.0) { return rand() * val / double(RAND_MAX); }
  static double r(double from, double to) { return rand() * (to - from) / double(RAND_MAX) + from; }
  static P4P r4(double m = -1) {
    double mass = (m != -1 ? m : (r() > .3 ? r(.1, 30) : 0));
    return P4P(r(5, 25), r(-2.5, 5), r(-M_PI, M_PI), mass);
  }
  static P4P p4near(const P4P &p, double d = 0.2) {
    double mass = (p.mass() == 0 ? 0 : p.mass() * r(1 - d, 1 + d));
    return P4P(p.pt() * r(1 - d, 1 + d), p.eta() + r(-d, +d), p.phi() + r(-d, +d), mass);
  }
  static V4 v4near(const V4 &v4, double d = 0.2) {
    V4 ret;
    ret[0] = v4[0] * r(1 - d, 1 + d);
    ret[1] = v4[1] * r(1 - d, 1 + d);
    ret[2] = v4[2] * r(1 - d, 1 + d);
    ret[3] = v4[3] * r(1 - d, 1 + d);
    return ret;
  }
  static P4C r4c(double m = -1) { return P4C(r4()); }
  static bool d(double x, double y, double eps = 1e-6) {
    return std::abs(x - y) < eps * (1 + std::abs(x) + std::abs(y));
  }
  static bool d4(P4P &p1, P4P &p2, double eps = 1e-6) {
    return d(p1.E(), p2.E(), eps) && d(p1.Px(), p2.Px(), eps) && d(p1.Py(), p2.Py(), eps) && d(p1.Pz(), p2.Pz(), eps);
  }
  static bool d4(P4C &p1, P4C &p2, double eps = 1e-6) {
    return d(p1.Pt(), p2.Pt(), eps) && d(p1.Eta(), p2.Eta(), eps) && d(p1.M(), p2.M(), eps) &&
           (deltaPhi(p1.Phi(), p2.Py()) < eps);
  }
  static bool d4(const V4 &v1, const V4 &v2, double eps = 1e-6) {
    return d(v1[0], v2[0], eps) && d(v1[1], v2[1], eps) && d(v1[2], v2[2], eps) && d(v1[3], v2[3], eps);
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION(testKinParametrizations);

bool testKinParametrizations::testTrivialVec2Par(Parametrization par, int tries) {
  using namespace pat::helper::ParametrizationHelper;
  srand(37);
  for (int i = 0; i < tries; ++i) {
    P4P p1 = r4(isAlwaysMassless(par) ? 0 : (r() > .3 ? r(.1, 30) : 0));
    V4 v4 = parametersFromP4(par, p1);
    P4P p2 = polarP4fromParameters(par, v4, p1);
    if (!d4(p1, p2)) {
      std::cerr << "\nFailure: (try " << i << "):"
                << "\n  p1 = " << p1 << "\n  p2 = " << p2 << "\n  v4 = " << v4 << std::endl;
      return false;
    }
  }
  return true;
}

bool testKinParametrizations::testVecVec2Diff(Parametrization par, int tries) {
  using namespace pat::helper::ParametrizationHelper;
  srand(37);
  for (int i = 0; i < tries; ++i) {
    P4P p1 = r4(isAlwaysMassless(par) ? 0 : (isAlwaysMassive(par) ? r(.1, 30) : (r() > .3 ? r(.1, 30) : 0)));
    V4 v1 = parametersFromP4(par, p1), v2;
    int phystries = 0;
    do {
      v2 = v4near(v1);
      if (dimension(par) == 3)
        v2[3] = v1[3];  // keep constraint
      phystries++;
    } while (!isPhysical(par, v2, p1) && (phystries < 50));
    if (!isPhysical(par, v2, p1)) {
      std::cerr << "\nFailure (try " << i << "/" << phystries << "):"
                << "\n  p1  = " << p1 << "\n  v1  = " << v1 << "\n  v2  = " << v2 << std::endl;
      return false;
    }
    V4 dv1 = v2 - v1;
    P4P p2 = polarP4fromParameters(par, v2, p1);
    V4 dv2 = diffToParameters(par, p1, p2);
    if (!d4(dv1, dv2)) {
      std::cerr << "\nFailure (try " << i << "):"
                << "\n  p1  = " << p1 << "\n  v1  = " << v1 << "\n  dv1 = " << dv1 << "\n  v2  = " << v2
                << "\n  p2  = " << p2 << "\n  dv2 = " << dv2 << std::endl;
      return false;
    }
  }
  return true;
}

bool testKinParametrizations::testVecDiff2Par(Parametrization par, int tries) {
  using namespace pat::helper::ParametrizationHelper;
  srand(37);
  for (int i = 0; i < tries; ++i) {
    P4P p1 = r4(isAlwaysMassless(par) ? 0 : (isAlwaysMassive(par) ? r(.1, 30) : (r() > .3 ? r(.1, 30) : 0)));
    V4 v1 = parametersFromP4(par, p1);
    P4P p2 = p4near(p1);
    if (isMassConstrained(par)) {
      p2.SetM(p1.mass());
    } else if (par == pat::CandKinResolution::EScaledMomDev) {
      // This is more tricky to get
      p2 = P4C(p2.px(), p2.py(), p2.pz(), p1.E() / p1.P() * p2.P());
    }
    V4 dv = diffToParameters(par, p1, p2);
    V4 v2 = v1 + dv;
    P4P p3 = polarP4fromParameters(par, v2, p1);
    if (!d4(p3, p2)) {
      std::cerr << "\nFailure: (try " << i << "):"
                << "\n  p1 = " << p1 << "\n  v1 = " << v1 << "\n  p2 = " << p2 << "\n  dv = " << dv << "\n  v2 = " << v2
                << "\n  p3 = " << p3 << std::endl;
      return false;
    }
  }
  return true;
}

void testKinParametrizations::testTrivialVec2Par_Cart() {
  CPPUNIT_ASSERT(testTrivialVec2Par(pat::CandKinResolution::Cart));
}
void testKinParametrizations::testTrivialVec2Par_ECart() {
  CPPUNIT_ASSERT(testTrivialVec2Par(pat::CandKinResolution::ECart));
}
void testKinParametrizations::testTrivialVec2Par_Spher() {
  CPPUNIT_ASSERT(testTrivialVec2Par(pat::CandKinResolution::Spher));
}
void testKinParametrizations::testTrivialVec2Par_ESpher() {
  CPPUNIT_ASSERT(testTrivialVec2Par(pat::CandKinResolution::ESpher));
}
void testKinParametrizations::testTrivialVec2Par_MomDev() {
  CPPUNIT_ASSERT(testTrivialVec2Par(pat::CandKinResolution::MomDev));
}
void testKinParametrizations::testTrivialVec2Par_EMomDev() {
  CPPUNIT_ASSERT(testTrivialVec2Par(pat::CandKinResolution::EMomDev));
}
void testKinParametrizations::testTrivialVec2Par_MCCart() {
  CPPUNIT_ASSERT(testTrivialVec2Par(pat::CandKinResolution::MCCart));
}
void testKinParametrizations::testTrivialVec2Par_MCSpher() {
  CPPUNIT_ASSERT(testTrivialVec2Par(pat::CandKinResolution::MCSpher));
}
void testKinParametrizations::testTrivialVec2Par_MCPInvSpher() {
  CPPUNIT_ASSERT(testTrivialVec2Par(pat::CandKinResolution::MCPInvSpher));
}
void testKinParametrizations::testTrivialVec2Par_EtEtaPhi() {
  CPPUNIT_ASSERT(testTrivialVec2Par(pat::CandKinResolution::EtEtaPhi));
}
void testKinParametrizations::testTrivialVec2Par_EtThetaPhi() {
  CPPUNIT_ASSERT(testTrivialVec2Par(pat::CandKinResolution::EtThetaPhi));
}
void testKinParametrizations::testTrivialVec2Par_MCMomDev() {
  CPPUNIT_ASSERT(testTrivialVec2Par(pat::CandKinResolution::MCMomDev));
}
void testKinParametrizations::testTrivialVec2Par_EScaledMomDev() {
  CPPUNIT_ASSERT(testTrivialVec2Par(pat::CandKinResolution::EScaledMomDev));
}

void testKinParametrizations::testVecDiff2Par_Cart() { CPPUNIT_ASSERT(testVecDiff2Par(pat::CandKinResolution::Cart)); }
void testKinParametrizations::testVecDiff2Par_ECart() {
  CPPUNIT_ASSERT(testVecDiff2Par(pat::CandKinResolution::ECart));
}
void testKinParametrizations::testVecDiff2Par_Spher() {
  CPPUNIT_ASSERT(testVecDiff2Par(pat::CandKinResolution::Spher));
}
void testKinParametrizations::testVecDiff2Par_ESpher() {
  CPPUNIT_ASSERT(testVecDiff2Par(pat::CandKinResolution::ESpher));
}
void testKinParametrizations::testVecDiff2Par_MCCart() {
  CPPUNIT_ASSERT(testVecDiff2Par(pat::CandKinResolution::MCCart));
}
void testKinParametrizations::testVecDiff2Par_MCSpher() {
  CPPUNIT_ASSERT(testVecDiff2Par(pat::CandKinResolution::MCSpher));
}
void testKinParametrizations::testVecDiff2Par_MCPInvSpher() {
  CPPUNIT_ASSERT(testVecDiff2Par(pat::CandKinResolution::MCPInvSpher));
}
void testKinParametrizations::testVecDiff2Par_EtEtaPhi() {
  CPPUNIT_ASSERT(testVecDiff2Par(pat::CandKinResolution::EtEtaPhi));
}
void testKinParametrizations::testVecDiff2Par_EtThetaPhi() {
  CPPUNIT_ASSERT(testVecDiff2Par(pat::CandKinResolution::EtThetaPhi));
}
void testKinParametrizations::testVecDiff2Par_MomDev() {
  CPPUNIT_ASSERT(testVecDiff2Par(pat::CandKinResolution::MomDev));
}
void testKinParametrizations::testVecDiff2Par_EMomDev() {
  CPPUNIT_ASSERT(testVecDiff2Par(pat::CandKinResolution::EMomDev));
}
void testKinParametrizations::testVecDiff2Par_MCMomDev() {
  CPPUNIT_ASSERT(testVecDiff2Par(pat::CandKinResolution::MCMomDev));
}
void testKinParametrizations::testVecDiff2Par_EScaledMomDev() {
  CPPUNIT_ASSERT(testVecDiff2Par(pat::CandKinResolution::EScaledMomDev));
}

void testKinParametrizations::testVecVec2Diff_Cart() { CPPUNIT_ASSERT(testVecVec2Diff(pat::CandKinResolution::Cart)); }
void testKinParametrizations::testVecVec2Diff_ECart() {
  CPPUNIT_ASSERT(testVecVec2Diff(pat::CandKinResolution::ECart));
}
void testKinParametrizations::testVecVec2Diff_Spher() {
  CPPUNIT_ASSERT(testVecVec2Diff(pat::CandKinResolution::Spher));
}
void testKinParametrizations::testVecVec2Diff_ESpher() {
  CPPUNIT_ASSERT(testVecVec2Diff(pat::CandKinResolution::ESpher));
}
void testKinParametrizations::testVecVec2Diff_MCCart() {
  CPPUNIT_ASSERT(testVecVec2Diff(pat::CandKinResolution::MCCart));
}
void testKinParametrizations::testVecVec2Diff_MCSpher() {
  CPPUNIT_ASSERT(testVecVec2Diff(pat::CandKinResolution::MCSpher));
}
void testKinParametrizations::testVecVec2Diff_MCPInvSpher() {
  CPPUNIT_ASSERT(testVecVec2Diff(pat::CandKinResolution::MCPInvSpher));
}
void testKinParametrizations::testVecVec2Diff_EtEtaPhi() {
  CPPUNIT_ASSERT(testVecVec2Diff(pat::CandKinResolution::EtEtaPhi));
}
void testKinParametrizations::testVecVec2Diff_EtThetaPhi() {
  CPPUNIT_ASSERT(testVecVec2Diff(pat::CandKinResolution::EtThetaPhi));
}
void testKinParametrizations::testVecVec2Diff_MomDev() {
  CPPUNIT_ASSERT(testVecVec2Diff(pat::CandKinResolution::MomDev));
}
void testKinParametrizations::testVecVec2Diff_EMomDev() {
  CPPUNIT_ASSERT(testVecVec2Diff(pat::CandKinResolution::EMomDev));
}
void testKinParametrizations::testVecVec2Diff_MCMomDev() {
  CPPUNIT_ASSERT(testVecVec2Diff(pat::CandKinResolution::MCMomDev));
}
void testKinParametrizations::testVecVec2Diff_EScaledMomDev() {
  CPPUNIT_ASSERT(testVecVec2Diff(pat::CandKinResolution::EScaledMomDev));
}
