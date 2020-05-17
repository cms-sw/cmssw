#include <cppunit/extensions/HelperMacros.h>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <TMath.h>

#include "DataFormats/PatCandidates/interface/ParametrizationHelper.h"
#include "DataFormats/PatCandidates/interface/ResolutionHelper.h"

class testKinResolutions : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testKinResolutions);

  CPPUNIT_TEST(testTrivialMatrix_Cart);
  CPPUNIT_TEST(testTrivialMatrix_ECart);
  CPPUNIT_TEST(testTrivialMatrix_Spher);
  CPPUNIT_TEST(testTrivialMatrix_ESpher);
  CPPUNIT_TEST(testTrivialMatrix_MomDev);
  CPPUNIT_TEST(testTrivialMatrix_EMomDev);
  CPPUNIT_TEST(testTrivialMatrix_MCCart);
  CPPUNIT_TEST(testTrivialMatrix_MCSpher);
  CPPUNIT_TEST(testTrivialMatrix_MCPInvSpher);
  CPPUNIT_TEST(testTrivialMatrix_EtEtaPhi);
  CPPUNIT_TEST(testTrivialMatrix_EtThetaPhi);
  CPPUNIT_TEST(testTrivialMatrix_MCMomDev);
  CPPUNIT_TEST(testTrivialMatrix_EScaledMomDev);

  CPPUNIT_TEST(testIndependentVars_Cart);
  CPPUNIT_TEST(testIndependentVars_ECart);
  CPPUNIT_TEST(testIndependentVars_Spher);
  CPPUNIT_TEST(testIndependentVars_ESpher);
  CPPUNIT_TEST(testIndependentVars_MomDev);
  CPPUNIT_TEST(testIndependentVars_EMomDev);
  CPPUNIT_TEST(testIndependentVars_MCCart);
  CPPUNIT_TEST(testIndependentVars_MCSpher);
  CPPUNIT_TEST(testIndependentVars_MCPInvSpher);
  CPPUNIT_TEST(testIndependentVars_EtEtaPhi);
  CPPUNIT_TEST(testIndependentVars_EtThetaPhi);
  CPPUNIT_TEST(testIndependentVars_MCMomDev);
  CPPUNIT_TEST(testIndependentVars_EScaledMomDev);

  CPPUNIT_TEST(testDependentVars_Cart);
  CPPUNIT_TEST(testDependentVars_ECart);
  CPPUNIT_TEST(testDependentVars_Spher);
  CPPUNIT_TEST(testDependentVars_ESpher);
  CPPUNIT_TEST(testDependentVars_MomDev);
  CPPUNIT_TEST(testDependentVars_EMomDev);
  CPPUNIT_TEST(testDependentVars_MCCart);
  CPPUNIT_TEST(testDependentVars_MCSpher);
  CPPUNIT_TEST(testDependentVars_MCPInvSpher);
  CPPUNIT_TEST(testDependentVars_EtEtaPhi);
  CPPUNIT_TEST(testDependentVars_EtThetaPhi);
  CPPUNIT_TEST(testDependentVars_MCMomDev);
  CPPUNIT_TEST(testDependentVars_EScaledMomDev);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}

  void testTrivialMatrix_Cart();
  void testTrivialMatrix_ECart();
  void testTrivialMatrix_Spher();
  void testTrivialMatrix_ESpher();
  void testTrivialMatrix_MomDev();
  void testTrivialMatrix_EMomDev();
  void testTrivialMatrix_MCCart();
  void testTrivialMatrix_MCSpher();
  void testTrivialMatrix_MCPInvSpher();
  void testTrivialMatrix_EtEtaPhi();
  void testTrivialMatrix_EtThetaPhi();
  void testTrivialMatrix_MCMomDev();
  void testTrivialMatrix_EScaledMomDev();

  void testIndependentVars_Cart();
  void testIndependentVars_ECart();
  void testIndependentVars_Spher();
  void testIndependentVars_ESpher();
  void testIndependentVars_MomDev();
  void testIndependentVars_EMomDev();
  void testIndependentVars_MCCart();
  void testIndependentVars_MCSpher();
  void testIndependentVars_MCPInvSpher();
  void testIndependentVars_EtEtaPhi();
  void testIndependentVars_EtThetaPhi();
  void testIndependentVars_MCMomDev();
  void testIndependentVars_EScaledMomDev();

  void testDependentVars_Cart();
  void testDependentVars_ECart();
  void testDependentVars_Spher();
  void testDependentVars_ESpher();
  void testDependentVars_MomDev();
  void testDependentVars_EMomDev();
  void testDependentVars_MCCart();
  void testDependentVars_MCSpher();
  void testDependentVars_MCPInvSpher();
  void testDependentVars_EtEtaPhi();
  void testDependentVars_EtThetaPhi();
  void testDependentVars_MCMomDev();
  void testDependentVars_EScaledMomDev();

  typedef math::XYZTLorentzVector P4C;
  typedef math::PtEtaPhiMLorentzVector P4P;
  typedef AlgebraicVector4 V4;
  typedef AlgebraicSymMatrix44 M4;

private:
  typedef pat::CandKinResolution::Parametrization Parametrization;

  /// check if resol 'f' for param 'p' throws an exception or not
  /// it won't even look at the value
  template <typename Func>
  bool testIfThrows(Parametrization p, Func f);

  /// check that resol 'f' for param 'p' is the square root of covariance(index,index)
  template <typename Func>
  bool testTrivialMatrix(Parametrization p, Func f, int index, int tries = 10);

  /// check that resol 'f' for param p is independent from the coordinate 'index'
  /// that is, if covariance(i,j) is null except for (index,index), resol 'f' must be zero
  template <typename Func>
  bool testIsIndependent(Parametrization p, Func f, int index, int tries = 10);

  /// check that resol 'f' for param p is independent from any coordinate index except 'self'
  template <typename Func>
  bool testFullyIndependent(Parametrization p, Func f, int self, int tries = 10);

  /// check the correctness of the derivative of some var the parameter, with respect to the numerical derivative
  /// computed symmetrically with step 'eps'.
  bool testDiagonalDerivativeM(Parametrization p, int index, double eps = 1e-5, int tries = 200);
  bool testDiagonalDerivativeEta(Parametrization p, int index, double eps = 1e-5, int tries = 200);
  bool testDiagonalDerivativeTheta(Parametrization p, int index, double eps = 1e-5, int tries = 200);
  bool testDiagonalDerivativePhi(Parametrization p, int index, double eps = 1e-5, int tries = 200);
  bool testDiagonalDerivativeEt(Parametrization p, int index, double eps = 1e-5, int tries = 200);
  bool testDiagonalDerivativeE(Parametrization p, int index, double eps = 1e-5, int tries = 200);
  bool testDiagonalDerivativeP(Parametrization p, int index, double eps = 1e-5, int tries = 200);
  bool testDiagonalDerivativePt(Parametrization p, int index, double eps = 1e-5, int tries = 200);
  bool testDiagonalDerivativePx(Parametrization p, int index, double eps = 1e-5, int tries = 200);
  bool testDiagonalDerivativePy(Parametrization p, int index, double eps = 1e-5, int tries = 200);
  bool testDiagonalDerivativePz(Parametrization p, int index, double eps = 1e-5, int tries = 200);
  bool testDiagonalDerivativePInv(Parametrization p, int index, double eps = 1e-5, int tries = 200);

  /// check the correctness of the relative signs of two derivative of some var the parameter
  bool testOffDiagonalDerivativeM(Parametrization p, int i, int j, double eps = 1e-5, int tries = 200);
  bool testOffDiagonalDerivativeEta(Parametrization p, int i, int j, double eps = 1e-5, int tries = 200);
  bool testOffDiagonalDerivativeTheta(Parametrization p, int i, int j, double eps = 1e-5, int tries = 200);
  bool testOffDiagonalDerivativePhi(Parametrization p, int i, int j, double eps = 1e-5, int tries = 200);
  bool testOffDiagonalDerivativeEt(Parametrization p, int i, int j, double eps = 1e-5, int tries = 200);
  bool testOffDiagonalDerivativeE(Parametrization p, int i, int j, double eps = 1e-5, int tries = 200);
  bool testOffDiagonalDerivativeP(Parametrization p, int i, int j, double eps = 1e-5, int tries = 200);
  bool testOffDiagonalDerivativePt(Parametrization p, int i, int j, double eps = 1e-5, int tries = 200);
  bool testOffDiagonalDerivativePx(Parametrization p, int i, int j, double eps = 1e-5, int tries = 200);
  bool testOffDiagonalDerivativePy(Parametrization p, int i, int j, double eps = 1e-5, int tries = 200);
  bool testOffDiagonalDerivativePz(Parametrization p, int i, int j, double eps = 1e-5, int tries = 200);
  bool testOffDiagonalDerivativePInv(Parametrization p, int i, int j, double eps = 1e-5, int tries = 200);

  double getVarEta(P4P &p4) const { return p4.Eta(); }
  double getVarTheta(P4P &p4) const { return p4.Theta(); }
  double getVarPhi(P4P &p4) const { return p4.Phi(); }
  double getVarEt(P4P &p4) const { return p4.Et(); }
  double getVarE(P4P &p4) const { return p4.E(); }
  double getVarP(P4P &p4) const { return p4.P(); }
  double getVarPt(P4P &p4) const { return p4.Pt(); }
  double getVarPx(P4P &p4) const { return p4.Px(); }
  double getVarPy(P4P &p4) const { return p4.Py(); }
  double getVarPz(P4P &p4) const { return p4.Pz(); }
  double getVarM(P4P &p4) const { return p4.M(); }
  double getVarPInv(P4P &p4) const { return 1.0 / p4.P(); }

  // Utilities
  static double r(double val = 1.0) { return rand() * val / double(RAND_MAX); }
  static double r(double from, double to) { return rand() * (to - from) / double(RAND_MAX) + from; }
  static P4P r4(double m = -1) {
    double mass = (m != -1 ? m : (r() > .3 ? r(.1, 30) : 0));
    return P4P(r(5, 25), r(-2.5, 5), r(-M_PI, M_PI), mass);
  }
  static M4 diag(size_t dim) {
    M4 ret;
    for (size_t i = 0; i < dim; ++i)
      ret(i, i) = r(.1, 10);
    return ret;
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION(testKinResolutions);

template <typename Func>
bool testKinResolutions::testIfThrows(Parametrization p, Func f) {
  srand(37);
  P4C p4;
  M4 mat = diag(pat::helper::ParametrizationHelper::dimension(p));
  double y = f(p, mat, p4);
  return (y != 1e37);  // so that it doesn't count as unused
}

using pat::helper::ParametrizationHelper::name;

template <typename Func>
bool testKinResolutions::testTrivialMatrix(Parametrization p, Func f, int index, int tries) {
  srand(37);
  using namespace pat::helper::ParametrizationHelper;
  for (int i = 0; i < tries; ++i) {
    P4C p4;
    M4 mat = diag(dimension(p));
    double x = sqrt(mat(index, index));
    double y = f(p, mat, p4);
    if (std::abs(x - y) > (std::abs(x) + std::abs(y) + 1) * 1e-6) {
      std::cerr << "Error for parametrization " << name(p) << ", index = " << index << "\n x = " << x << ", y = " << y
                << std::endl;
      return false;
    }
  }
  return true;
}

template <typename Func>
bool testKinResolutions::testIsIndependent(Parametrization par, Func f, int index, int tries) {
  double delta = 0.001;
  using namespace pat::helper::ParametrizationHelper;
  srand(37);
  for (int i = 0; i < tries; ++i) {
    P4P p1 = r4(isAlwaysMassless(par) ? 0 : (isAlwaysMassive(par) ? r(.1, 30) : (r() > .3 ? r(.1, 30) : 0)));
    //         V4  v1  = parametersFromP4(par, p1), v2; // warning from gcc461: variable 'v1' set but not used [-Wunused-but-set-variable]
    M4 mat;
    mat(index, index) = delta * delta;
    double exp = f(par, mat, P4C(p1));
    if (exp != 0)
      return false;
  }
  return true;
}
template <typename Func>
bool testKinResolutions::testFullyIndependent(Parametrization par, Func f, int self, int tries) {
  for (int i = 0; i <= 3; ++i) {
    if (i == self)
      continue;
    if (!testIsIndependent(par, f, i, tries))
      return false;
  }
  return true;
}

#define IMPL_testDiagonalDerivative(VAR)                                                                            \
  bool testKinResolutions::testDiagonalDerivative##VAR(Parametrization par, int index, double delta, int tries) {   \
    using namespace pat::helper::ParametrizationHelper;                                                             \
    using namespace pat::helper::ResolutionHelper;                                                                  \
    srand(37);                                                                                                      \
    int skips = 0, glitches = 0;                                                                                    \
    for (int i = 0; i < tries; ++i) {                                                                               \
      P4P p1 = r4(isAlwaysMassless(par) ? 0 : r(.1, 30));                                                           \
      M4 mat;                                                                                                       \
      mat(index, index) = 1;                                                                                        \
      double exp = getResol##VAR(                                                                                   \
          par, mat, P4C(p1)); /* should be |dV/dX_i|, where V is the variable on which 'f' gets the resolution */   \
      /* now let's compute the derivative numerically */                                                            \
      V4 v = parametersFromP4(par, p1), vplus = v, vminus = v;                                                      \
      vplus[index] += delta;                                                                                        \
      vminus[index] -= delta;                                                                                       \
      if (isPhysical(par, vplus, p1) && (isPhysical(par, vminus, p1))) {                                            \
        P4P pplus = polarP4fromParameters(par, vplus, p1);                                                          \
        P4P pminus = polarP4fromParameters(par, vminus, p1);                                                        \
        double yplus = getVar##VAR(pplus), yminus = getVar##VAR(pminus);                                            \
        double num = (yplus - yminus) / (2 * delta);                                                                \
        double diff = std::abs(std::abs(num) - exp) / (std::abs(num) + std::abs(exp) + 1);                          \
        if (diff > 2e-4) {                                                                                          \
          std::cout << "\nError for  " #VAR "\n"                                                                    \
                    << "  par = " << name(par) << ", index = " << index << "\n"                                     \
                    << "  p4 = " << p1 << ",\n  expected = " << exp << ", numeric = " << num << ", diff = " << diff \
                    << std::endl;                                                                                   \
          glitches++;                                                                                               \
          if ((diff > 3e-2) || (glitches > std::max(tries / 10, 10)))                                               \
            return false;                                                                                           \
        }                                                                                                           \
      } else                                                                                                        \
        skips++;                                                                                                    \
    }                                                                                                               \
    if (double(skips) / tries >= 0.1) {                                                                             \
      std::cout << "Error for  " #VAR "\n"                                                                          \
                << "  par = " << name(par) << ", index = " << index                                                 \
                << "\n"                                                                                             \
                   "Unphysical momenta "                                                                            \
                << skips << " over " << tries << std::endl;                                                         \
    }                                                                                                               \
    return (double(skips) / tries < 0.1);                                                                           \
  }
IMPL_testDiagonalDerivative(M) IMPL_testDiagonalDerivative(Eta) IMPL_testDiagonalDerivative(Theta)
    IMPL_testDiagonalDerivative(Phi) IMPL_testDiagonalDerivative(Et) IMPL_testDiagonalDerivative(E)
        IMPL_testDiagonalDerivative(P) IMPL_testDiagonalDerivative(Pt) IMPL_testDiagonalDerivative(Px)
            IMPL_testDiagonalDerivative(Py) IMPL_testDiagonalDerivative(Pz) IMPL_testDiagonalDerivative(PInv)

#define IMPL_testOffDiagonalDerivative(VAR)                                                                         \
  bool testKinResolutions::testOffDiagonalDerivative##VAR(                                                          \
      Parametrization par, int i, int j, double delta, int tries) {                                                 \
    using namespace pat::helper::ParametrizationHelper;                                                             \
    using namespace pat::helper::ResolutionHelper;                                                                  \
    srand(37);                                                                                                      \
    int skips = 0, glitches = 0;                                                                                    \
    for (int it = 0; it < tries; ++it) {                                                                            \
      P4P p1 = r4(isAlwaysMassless(par) ? 0 : r(.1, 30));                                                           \
      M4 mat;                                                                                                       \
      mat(i, i) = 1;                                                                                                \
      double vi = getResol##VAR(                                                                                    \
          par, mat, P4C(p1)); /* should be |dV/dX_i|, where V is the variable on which 'f' gets the resolution */   \
      mat(i, i) = 0;                                                                                                \
      mat(j, j) = 1;                                                                                                \
      double vj = getResol##VAR(                                                                                    \
          par, mat, P4C(p1)); /* should be |dV/dX_j|, where V is the variable on which 'f' gets the resolution */   \
      mat(i, i) = 2;                                                                                                \
      mat(j, j) = 2;                                                                                                \
      mat(i, j) = 1;                                                                                                \
      double vij =                                                                                                  \
          getResol##VAR(par, mat, P4C(p1)); /* should be sqrt(2)*(|dV/dX_i|^2+|dV/dx_j|^2 + 2 dV/dX_i dV_dX_j ) */  \
      /* now let's compute the derivative numerically */                                                            \
      V4 v = parametersFromP4(par, p1), vplus_i = v, vminus_i = v, vplus_j = v, vminus_j = v;                       \
      vplus_i[i] += delta;                                                                                          \
      vminus_i[i] -= delta;                                                                                         \
      vplus_j[j] += delta;                                                                                          \
      vminus_j[j] -= delta;                                                                                         \
      if (isPhysical(par, vplus_i, p1) && (isPhysical(par, vminus_i, p1)) && isPhysical(par, vplus_j, p1) &&        \
          (isPhysical(par, vminus_j, p1))) {                                                                        \
        P4P pplus_i = polarP4fromParameters(par, vplus_i, p1);                                                      \
        P4P pplus_j = polarP4fromParameters(par, vplus_j, p1);                                                      \
        P4P pminus_i = polarP4fromParameters(par, vminus_i, p1);                                                    \
        P4P pminus_j = polarP4fromParameters(par, vminus_j, p1);                                                    \
        double yplus_i = getVar##VAR(pplus_i), yminus_i = getVar##VAR(pminus_i);                                    \
        double yplus_j = getVar##VAR(pplus_j), yminus_j = getVar##VAR(pminus_j);                                    \
        double num_i = (yplus_i - yminus_i) / (2 * delta);                                                          \
        double num_j = (yplus_j - yminus_j) / (2 * delta);                                                          \
        if (num_i < 0)                                                                                              \
          vi = -vi;                                                                                                 \
        if (num_j < 0)                                                                                              \
          vj = -vj;                                                                                                 \
        double num = std::sqrt(2 * (vi * vi + vj * vj + vi * vj));                                                  \
        double diff = std::abs(std::abs(num) - vij) / (std::abs(num) + std::abs(vij) + 1);                          \
        if (diff > 1e-4) {                                                                                          \
          std::cout << "\nError for  " #VAR "\n"                                                                    \
                    << "  par = " << name(par) << ", i = " << i << ", j= " << j << "\n"                             \
                    << "  p4 = " << p1 << ",\n  expected = " << vij << ", numeric = " << num << ", diff = " << diff \
                    << std::endl;                                                                                   \
          glitches++;                                                                                               \
          if ((diff > 3e-2) || (glitches > std::max(tries / 10, 10)))                                               \
            return false;                                                                                           \
        }                                                                                                           \
      } else                                                                                                        \
        skips++;                                                                                                    \
    }                                                                                                               \
    if (double(skips) / tries >= 0.1) {                                                                             \
      std::cout << "Error for  " #VAR "\n"                                                                          \
                << "  par = " << name(par) << ", i = " << i << ", j= " << j                                         \
                << "\n"                                                                                             \
                   "Unphysical momenta "                                                                            \
                << skips << " over " << tries << std::endl;                                                         \
    }                                                                                                               \
    return (double(skips) / tries < 0.1);                                                                           \
  }
                IMPL_testOffDiagonalDerivative(M) IMPL_testOffDiagonalDerivative(Eta)
                    IMPL_testOffDiagonalDerivative(Theta) IMPL_testOffDiagonalDerivative(Phi)
                        IMPL_testOffDiagonalDerivative(Et) IMPL_testOffDiagonalDerivative(E)
                            IMPL_testOffDiagonalDerivative(P) IMPL_testOffDiagonalDerivative(Pt)
                                IMPL_testOffDiagonalDerivative(Px) IMPL_testOffDiagonalDerivative(Py)
                                    IMPL_testOffDiagonalDerivative(Pz) IMPL_testOffDiagonalDerivative(PInv)

#define ASSERT_DIAGONAL(PARAM, RESOL, INDEX)                                                                    \
  if (!testTrivialMatrix(pat::CandKinResolution::PARAM, RESOL, INDEX)) {                                        \
    CPPUNIT_NS::Asserter::fail("Resolution " #RESOL " for Parametrization " #PARAM " is not covariance(" #INDEX \
                               "," #INDEX ")",                                                                  \
                               CPPUNIT_SOURCELINE());                                                           \
  }

#define ASSERT_FULLY_INDEPENDENT(PARAM, RESOL, INDEX)                                                                  \
  if (!testFullyIndependent(pat::CandKinResolution::PARAM, RESOL, INDEX)) {                                            \
    CPPUNIT_NS::Asserter::fail("Resolution " #RESOL " for Parametrization " #PARAM " is not only function of " #INDEX, \
                               CPPUNIT_SOURCELINE());                                                                  \
  }

// Derivatives must not throw exceptions
#define ASSERT_HAS_DERIVATIVE(PARAM, RESOL) CPPUNIT_ASSERT_NO_THROW(testIfThrows(pat::CandKinResolution::PARAM, RESOL));
// And we can check their independence from some coordinate
#define ASSERT_HAS_INDEP_DERIVATIVE(PARAM, RESOL, INDEX)                                                        \
  if (!testIsIndependent(pat::CandKinResolution::PARAM, RESOL, INDEX)) {                                        \
    CPPUNIT_NS::Asserter::fail("Resolution " #RESOL " for Parametrization " #PARAM " is correlated to " #INDEX, \
                               CPPUNIT_SOURCELINE());                                                           \
  }
#define ASSERT_CHECK_DERIVATIVE(PARAM, VAR, INDEX)                                  \
  if (!testDiagonalDerivative##VAR(pat::CandKinResolution::PARAM, INDEX)) {         \
    CPPUNIT_NS::Asserter::fail("Resolution on " #VAR " for Parametrization " #PARAM \
                               " has wrong derivative w.r.t. " #INDEX,              \
                               CPPUNIT_SOURCELINE());                               \
  }
#define ASSERT_CHECK_DERIVATIVE2(PARAM, VAR, I, J)                                                                     \
  if (!testOffDiagonalDerivative##VAR(pat::CandKinResolution::PARAM, I, J)) {                                          \
    CPPUNIT_NS::Asserter::fail("Resolution on " #VAR " for Parametrization " #PARAM " has wrong derivative w.r.t. " #I \
                               "," #J,                                                                                 \
                               CPPUNIT_SOURCELINE());                                                                  \
  }

#define ASSERT_CHECK_DERIVATIVES(PARAM, VAR)           \
  ASSERT_CHECK_DERIVATIVE(PARAM, VAR, 0)               \
  ASSERT_CHECK_DERIVATIVE(PARAM, VAR, 1)               \
  ASSERT_CHECK_DERIVATIVE(PARAM, VAR, 2)               \
  ASSERT_CHECK_DERIVATIVE2(PARAM, VAR, 0, 1)           \
  ASSERT_CHECK_DERIVATIVE2(PARAM, VAR, 0, 2)           \
  ASSERT_CHECK_DERIVATIVE2(PARAM, VAR, 1, 2)           \
  if (dimension(pat::CandKinResolution::PARAM) == 4) { \
    ASSERT_CHECK_DERIVATIVE(PARAM, VAR, 3)             \
    ASSERT_CHECK_DERIVATIVE2(PARAM, VAR, 0, 3)         \
    ASSERT_CHECK_DERIVATIVE2(PARAM, VAR, 1, 3)         \
    ASSERT_CHECK_DERIVATIVE2(PARAM, VAR, 2, 3)         \
  }

#define ASSERT_NOT_IMPLEMENTED(PARAM, RESOL)                                       \
  do {                                                                             \
    bool cpputExceptionThrown_ = false;                                            \
    try {                                                                          \
      testTrivialMatrix(pat::CandKinResolution::PARAM, RESOL, 0, 1);               \
    } catch (const cms::Exception &e) {                                            \
      if (e.category() == "Not Implemented")                                       \
        cpputExceptionThrown_ = true;                                              \
    }                                                                              \
                                                                                   \
    if (cpputExceptionThrown_)                                                     \
      break;                                                                       \
                                                                                   \
    CPPUNIT_NS::Asserter::fail("Resolution " #RESOL " for Parametrization " #PARAM \
                               " Not implemented but not throwing",                \
                               CPPUNIT_SOURCELINE());                              \
  } while (false)

                                        using namespace pat::helper::ResolutionHelper;
using pat::helper::ParametrizationHelper::dimension;

void testKinResolutions::testTrivialMatrix_Cart() {
  ASSERT_DIAGONAL(Cart, getResolPx, 0);
  ASSERT_DIAGONAL(Cart, getResolPy, 1);
  ASSERT_DIAGONAL(Cart, getResolPz, 2);
  ASSERT_DIAGONAL(Cart, getResolM, 3);
  ASSERT_HAS_DERIVATIVE(Cart, getResolEta);
  ASSERT_HAS_DERIVATIVE(Cart, getResolTheta);
  ASSERT_HAS_DERIVATIVE(Cart, getResolPhi);
  ASSERT_HAS_DERIVATIVE(Cart, getResolEt);
  ASSERT_HAS_DERIVATIVE(Cart, getResolE);
  ASSERT_HAS_DERIVATIVE(Cart, getResolP);
  ASSERT_HAS_DERIVATIVE(Cart, getResolPt);
  ASSERT_HAS_DERIVATIVE(Cart, getResolPInv);
}
void testKinResolutions::testTrivialMatrix_ECart() {
  ASSERT_DIAGONAL(ECart, getResolPx, 0);
  ASSERT_DIAGONAL(ECart, getResolPy, 1);
  ASSERT_DIAGONAL(ECart, getResolPz, 2);
  ASSERT_DIAGONAL(ECart, getResolE, 3);
  ASSERT_HAS_DERIVATIVE(ECart, getResolEta);
  ASSERT_HAS_DERIVATIVE(ECart, getResolTheta);
  ASSERT_HAS_DERIVATIVE(ECart, getResolPhi);
  ASSERT_HAS_DERIVATIVE(ECart, getResolEt);
  ASSERT_HAS_DERIVATIVE(ECart, getResolM);
  ASSERT_HAS_DERIVATIVE(ECart, getResolP);
  ASSERT_HAS_DERIVATIVE(ECart, getResolPt);
  ASSERT_HAS_DERIVATIVE(ECart, getResolPInv);
}

void testKinResolutions::testTrivialMatrix_Spher() {
  ASSERT_DIAGONAL(Spher, getResolP, 0);
  ASSERT_DIAGONAL(Spher, getResolTheta, 1);
  ASSERT_DIAGONAL(Spher, getResolPhi, 2);
  ASSERT_DIAGONAL(Spher, getResolM, 3);
  ASSERT_HAS_DERIVATIVE(Spher, getResolEta);
  ASSERT_HAS_DERIVATIVE(Spher, getResolEt);
  ASSERT_HAS_DERIVATIVE(Spher, getResolE);
  ASSERT_HAS_DERIVATIVE(Spher, getResolPt);
  ASSERT_HAS_DERIVATIVE(Spher, getResolPInv);
  ASSERT_HAS_DERIVATIVE(Spher, getResolPx);
  ASSERT_HAS_DERIVATIVE(Spher, getResolPy);
  ASSERT_HAS_DERIVATIVE(Spher, getResolPz);
}

void testKinResolutions::testTrivialMatrix_ESpher() {
  ASSERT_DIAGONAL(ESpher, getResolP, 0);
  ASSERT_DIAGONAL(ESpher, getResolTheta, 1);
  ASSERT_DIAGONAL(ESpher, getResolPhi, 2);
  ASSERT_DIAGONAL(ESpher, getResolE, 3);
  ASSERT_HAS_DERIVATIVE(ESpher, getResolEta);
  ASSERT_HAS_DERIVATIVE(ESpher, getResolEt);
  ASSERT_HAS_DERIVATIVE(ESpher, getResolM);
  ASSERT_HAS_DERIVATIVE(ESpher, getResolPt);
  ASSERT_HAS_DERIVATIVE(ESpher, getResolPInv);
  ASSERT_HAS_DERIVATIVE(ESpher, getResolPx);
  ASSERT_HAS_DERIVATIVE(ESpher, getResolPy);
  ASSERT_HAS_DERIVATIVE(ESpher, getResolPz);
}

void testKinResolutions::testTrivialMatrix_MomDev() {
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolEta);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolTheta);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolPhi);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolEt);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolE);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolP);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolPt);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolPInv);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolPx);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolPy);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolPz);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolM);
}

void testKinResolutions::testTrivialMatrix_EMomDev() {
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolEta);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolTheta);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolPhi);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolEt);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolE);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolP);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolPt);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolPInv);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolPx);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolPy);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolPz);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolM);
}

void testKinResolutions::testTrivialMatrix_MCCart() {
  ASSERT_DIAGONAL(MCCart, getResolPx, 0);
  ASSERT_DIAGONAL(MCCart, getResolPy, 1);
  ASSERT_DIAGONAL(MCCart, getResolPz, 2);
  ASSERT_HAS_DERIVATIVE(MCCart, getResolEta);
  ASSERT_HAS_DERIVATIVE(MCCart, getResolTheta);
  ASSERT_HAS_DERIVATIVE(MCCart, getResolPhi);
  ASSERT_HAS_DERIVATIVE(MCCart, getResolEt);
  ASSERT_HAS_DERIVATIVE(MCCart, getResolE);
  ASSERT_HAS_DERIVATIVE(MCCart, getResolP);
  ASSERT_HAS_DERIVATIVE(MCCart, getResolPt);
  ASSERT_HAS_DERIVATIVE(MCCart, getResolPInv);
}

void testKinResolutions::testTrivialMatrix_MCSpher() {
  ASSERT_DIAGONAL(MCSpher, getResolP, 0);
  ASSERT_DIAGONAL(MCSpher, getResolTheta, 1);
  ASSERT_DIAGONAL(MCSpher, getResolPhi, 2);
  ASSERT_HAS_DERIVATIVE(MCSpher, getResolEta);
  ASSERT_HAS_DERIVATIVE(MCSpher, getResolEt);
  ASSERT_HAS_DERIVATIVE(MCSpher, getResolE);
  ASSERT_HAS_DERIVATIVE(MCSpher, getResolPt);
  ASSERT_HAS_DERIVATIVE(MCSpher, getResolPInv);
  ASSERT_HAS_DERIVATIVE(MCSpher, getResolPx);
  ASSERT_HAS_DERIVATIVE(MCSpher, getResolPy);
  ASSERT_HAS_DERIVATIVE(MCSpher, getResolPz);
}

void testKinResolutions::testTrivialMatrix_MCPInvSpher() {
  ASSERT_DIAGONAL(MCPInvSpher, getResolPInv, 0);
  ASSERT_DIAGONAL(MCPInvSpher, getResolTheta, 1);
  ASSERT_DIAGONAL(MCPInvSpher, getResolPhi, 2);
  ASSERT_HAS_DERIVATIVE(MCPInvSpher, getResolEta);
  ASSERT_HAS_DERIVATIVE(MCPInvSpher, getResolEt);
  ASSERT_HAS_DERIVATIVE(MCPInvSpher, getResolE);
  ASSERT_HAS_DERIVATIVE(MCPInvSpher, getResolP);
  ASSERT_HAS_DERIVATIVE(MCPInvSpher, getResolPt);
  ASSERT_HAS_DERIVATIVE(MCPInvSpher, getResolPx);
  ASSERT_HAS_DERIVATIVE(MCPInvSpher, getResolPy);
  ASSERT_HAS_DERIVATIVE(MCPInvSpher, getResolPz);
}

void testKinResolutions::testTrivialMatrix_EtEtaPhi() {
  ASSERT_DIAGONAL(EtEtaPhi, getResolEt, 0);
  ASSERT_DIAGONAL(EtEtaPhi, getResolPt, 0);  // Et == Pt
  ASSERT_DIAGONAL(EtEtaPhi, getResolEta, 1);
  ASSERT_DIAGONAL(EtEtaPhi, getResolPhi, 2);
  ASSERT_HAS_DERIVATIVE(EtEtaPhi, getResolTheta);
  ASSERT_HAS_DERIVATIVE(EtEtaPhi, getResolE);
  ASSERT_HAS_DERIVATIVE(EtEtaPhi, getResolP);
  ASSERT_HAS_DERIVATIVE(EtEtaPhi, getResolPInv);
  ASSERT_HAS_DERIVATIVE(EtEtaPhi, getResolPx);
  ASSERT_HAS_DERIVATIVE(EtEtaPhi, getResolPy);
  ASSERT_HAS_DERIVATIVE(EtEtaPhi, getResolPz);
}

void testKinResolutions::testTrivialMatrix_EtThetaPhi() {
  ASSERT_DIAGONAL(EtThetaPhi, getResolEt, 0);
  ASSERT_DIAGONAL(EtThetaPhi, getResolPt, 0);  // Et == Pt
  ASSERT_DIAGONAL(EtThetaPhi, getResolTheta, 1);
  ASSERT_DIAGONAL(EtThetaPhi, getResolPhi, 2);
  ASSERT_HAS_DERIVATIVE(EtThetaPhi, getResolEta);
  ASSERT_HAS_DERIVATIVE(EtThetaPhi, getResolE);
  ASSERT_HAS_DERIVATIVE(EtThetaPhi, getResolP);
  ASSERT_HAS_DERIVATIVE(EtThetaPhi, getResolPInv);
  ASSERT_HAS_DERIVATIVE(EtThetaPhi, getResolPx);
  ASSERT_HAS_DERIVATIVE(EtThetaPhi, getResolPy);
  ASSERT_HAS_DERIVATIVE(EtThetaPhi, getResolPz);
}

void testKinResolutions::testTrivialMatrix_MCMomDev() {
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolEta);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolTheta);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolPhi);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolEt);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolE);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolP);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolPt);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolPInv);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolPx);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolPy);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolPz);
}

void testKinResolutions::testTrivialMatrix_EScaledMomDev() {
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolEta);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolTheta);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolPhi);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolEt);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolE);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolP);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolPt);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolPInv);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolPx);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolPy);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolPz);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolM);
}

void testKinResolutions::testIndependentVars_Cart() {
  ASSERT_FULLY_INDEPENDENT(Cart, getResolPx, 0);
  ASSERT_FULLY_INDEPENDENT(Cart, getResolPy, 1);
  ASSERT_FULLY_INDEPENDENT(Cart, getResolPz, 2);
  ASSERT_FULLY_INDEPENDENT(Cart, getResolM, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(Cart, getResolEta, 3);    // Angles Don't
  ASSERT_HAS_INDEP_DERIVATIVE(Cart, getResolTheta, 3);  // depend on
  ASSERT_HAS_INDEP_DERIVATIVE(Cart, getResolPhi, 3);    // the mass
}
void testKinResolutions::testIndependentVars_ECart() {
  ASSERT_FULLY_INDEPENDENT(ECart, getResolPx, 0);
  ASSERT_FULLY_INDEPENDENT(ECart, getResolPy, 1);
  ASSERT_FULLY_INDEPENDENT(ECart, getResolPz, 2);
  ASSERT_FULLY_INDEPENDENT(ECart, getResolE, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(ECart, getResolEta, 3);    // Angles
  ASSERT_HAS_INDEP_DERIVATIVE(ECart, getResolTheta, 3);  // Don't depend
  ASSERT_HAS_INDEP_DERIVATIVE(ECart, getResolPhi, 3);    // on the energy
}

void testKinResolutions::testIndependentVars_Spher() {
  ASSERT_FULLY_INDEPENDENT(Spher, getResolP, 0);
  ASSERT_FULLY_INDEPENDENT(Spher, getResolTheta, 1);
  ASSERT_FULLY_INDEPENDENT(Spher, getResolPhi, 2);
  ASSERT_FULLY_INDEPENDENT(Spher, getResolM, 3);

  ASSERT_HAS_INDEP_DERIVATIVE(Spher, getResolEta, 0);  // Eta
  ASSERT_HAS_INDEP_DERIVATIVE(Spher, getResolEta, 2);  // Depends
  ASSERT_HAS_INDEP_DERIVATIVE(Spher, getResolEta, 3);  // Only On Theta

  ASSERT_HAS_INDEP_DERIVATIVE(Spher, getResolEt, 2);  // E, Et
  ASSERT_HAS_INDEP_DERIVATIVE(Spher, getResolE, 2);   // Don't depend on Phi

  ASSERT_HAS_INDEP_DERIVATIVE(Spher, getResolPt, 2);  // Pt indep from
  ASSERT_HAS_INDEP_DERIVATIVE(Spher, getResolPt, 3);  // Phi and M

  ASSERT_HAS_INDEP_DERIVATIVE(Spher, getResolPInv, 1);  // PInv dep
  ASSERT_HAS_INDEP_DERIVATIVE(Spher, getResolPInv, 2);  // Only on
  ASSERT_HAS_INDEP_DERIVATIVE(Spher, getResolPInv, 3);  // P

  ASSERT_HAS_INDEP_DERIVATIVE(Spher, getResolPx, 3);  // Px, Py, Pz
  ASSERT_HAS_INDEP_DERIVATIVE(Spher, getResolPy, 3);  // Indep from
  ASSERT_HAS_INDEP_DERIVATIVE(Spher, getResolPz, 3);  // Mass
  ASSERT_HAS_INDEP_DERIVATIVE(Spher, getResolPz, 3);  // Pz also on Phi
}

void testKinResolutions::testIndependentVars_ESpher() {
  ASSERT_FULLY_INDEPENDENT(ESpher, getResolP, 0);
  ASSERT_FULLY_INDEPENDENT(ESpher, getResolTheta, 1);
  ASSERT_FULLY_INDEPENDENT(ESpher, getResolPhi, 2);
  ASSERT_FULLY_INDEPENDENT(ESpher, getResolE, 3);

  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolEta, 0);  // Eta
  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolEta, 2);  // Depends
  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolEta, 3);  // Only On Theta

  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolEt, 2);  // Et indep from Phi
  //ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolEt,  2); // FIXME And from P??

  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolM, 2);  // Don't depend on Phi
  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolM, 1);  // Nor on Theta

  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolPt, 2);  // Pt indep from
  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolPt, 3);  // Phi and E

  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolPInv, 1);  // PInv dep
  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolPInv, 2);  // Only on
  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolPInv, 3);  // P

  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolPx, 3);  // Px, Py, Pz
  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolPy, 3);  // Indep from
  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolPz, 3);  // Energy
  ASSERT_HAS_INDEP_DERIVATIVE(ESpher, getResolPz, 3);  // Pz also on Phi
}

void testKinResolutions::testIndependentVars_MomDev() {
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolEta);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolTheta);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolPhi);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolEt);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolE);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolP);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolPt);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolPInv);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolPx);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolPy);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolPz);
  ASSERT_NOT_IMPLEMENTED(MomDev, getResolM);
}

void testKinResolutions::testIndependentVars_EMomDev() {
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolEta);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolTheta);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolPhi);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolEt);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolE);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolP);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolPt);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolPInv);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolPx);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolPy);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolPz);
  ASSERT_NOT_IMPLEMENTED(EMomDev, getResolM);
}

void testKinResolutions::testIndependentVars_MCCart() {
  ASSERT_FULLY_INDEPENDENT(MCCart, getResolPx, 0);
  ASSERT_FULLY_INDEPENDENT(MCCart, getResolPy, 1);
  ASSERT_FULLY_INDEPENDENT(MCCart, getResolPz, 2);

  ASSERT_HAS_INDEP_DERIVATIVE(MCCart, getResolPhi, 2);  // Phi doesn't depend on Pz

  ASSERT_HAS_INDEP_DERIVATIVE(MCCart, getResolEta, 3);    // nothing
  ASSERT_HAS_INDEP_DERIVATIVE(MCCart, getResolTheta, 3);  // depends
  ASSERT_HAS_INDEP_DERIVATIVE(MCCart, getResolPhi, 3);    // on M
  ASSERT_HAS_INDEP_DERIVATIVE(MCCart, getResolEt, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(MCCart, getResolE, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(MCCart, getResolP, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(MCCart, getResolPt, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(MCCart, getResolPInv, 3);
}

void testKinResolutions::testIndependentVars_MCSpher() {
  ASSERT_FULLY_INDEPENDENT(MCSpher, getResolP, 0);
  ASSERT_FULLY_INDEPENDENT(MCSpher, getResolTheta, 1);
  ASSERT_FULLY_INDEPENDENT(MCSpher, getResolPhi, 2);

  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolEta, 3);  // everything
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolEt, 3);   // indep from
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolE, 3);    // mass
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPt, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPInv, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPx, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPy, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPz, 3);

  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolEta, 2);  // Most things
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolEt, 2);   // Indep from Phi
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolE, 2);
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPt, 2);
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPInv, 2);
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPz, 2);

  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolE, 1);     // E, 1/P
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPInv, 1);  // dep only on P
}

void testKinResolutions::testIndependentVars_MCPInvSpher() {
  ASSERT_FULLY_INDEPENDENT(MCPInvSpher, getResolPInv, 0);
  ASSERT_FULLY_INDEPENDENT(MCPInvSpher, getResolTheta, 1);
  ASSERT_FULLY_INDEPENDENT(MCPInvSpher, getResolPhi, 2);

  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolEta, 3);  // everything
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolEt, 3);   // indep from
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolE, 3);    // mass
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPt, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolP, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPx, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPy, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPz, 3);

  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolEta, 2);  // Most things
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolEt, 2);   // Indep from Phi
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolE, 2);
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPt, 2);
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolP, 2);
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolPz, 2);

  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolE, 1);  // E, P
  ASSERT_HAS_INDEP_DERIVATIVE(MCSpher, getResolP, 1);  // dep only on 1/P
}

void testKinResolutions::testIndependentVars_EtEtaPhi() {
  ASSERT_FULLY_INDEPENDENT(EtEtaPhi, getResolEt, 0);
  ASSERT_FULLY_INDEPENDENT(EtEtaPhi, getResolPt, 0);  // Et == Pt
  ASSERT_FULLY_INDEPENDENT(EtEtaPhi, getResolEta, 1);
  ASSERT_FULLY_INDEPENDENT(EtEtaPhi, getResolPhi, 2);

  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolTheta, 3);  // All indep
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolE, 3);      // From the mass
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolP, 3);      // (which is also
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPInv, 3);   // always zero)
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPx, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPy, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPz, 3);

  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolTheta, 2);  // Most things
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolE, 2);      // Indep from
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolP, 2);      // Phi
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPInv, 2);
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPz, 2);

  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPx, 1);  // Px, Py
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPy, 1);  // Indep from Eta

  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolTheta, 0);  // Theta indep from Et
}

void testKinResolutions::testIndependentVars_EtThetaPhi() {
  ASSERT_FULLY_INDEPENDENT(EtThetaPhi, getResolEt, 0);
  ASSERT_FULLY_INDEPENDENT(EtThetaPhi, getResolPt, 0);  // Et == Pt
  ASSERT_FULLY_INDEPENDENT(EtThetaPhi, getResolTheta, 1);
  ASSERT_FULLY_INDEPENDENT(EtThetaPhi, getResolPhi, 2);

  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolEta, 3);   // All indep
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolE, 3);     // From the mass
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolP, 3);     // (which is also
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPInv, 3);  // always zero)
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPx, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPy, 3);
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPz, 3);

  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolEta, 2);  // Most things
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolE, 2);    // Indep from
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolP, 2);    // Phi
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPInv, 2);
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPz, 2);

  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPx, 1);  // Px, Py
  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolPy, 1);  // Indep from Theta

  ASSERT_HAS_INDEP_DERIVATIVE(EtEtaPhi, getResolEta, 0);  // Eta indep from Et
}

void testKinResolutions::testIndependentVars_MCMomDev() {
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolEta);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolTheta);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolPhi);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolEt);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolE);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolP);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolPt);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolPInv);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolPx);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolPy);
  ASSERT_NOT_IMPLEMENTED(MCMomDev, getResolPz);
}

void testKinResolutions::testIndependentVars_EScaledMomDev() {
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolEta);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolTheta);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolPhi);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolEt);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolE);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolP);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolPt);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolPInv);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolPx);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolPy);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolPz);
  ASSERT_NOT_IMPLEMENTED(EScaledMomDev, getResolM);
}

#define IMPL_testDependentVars(PAR)                    \
  void testKinResolutions::testDependentVars_##PAR() { \
    ASSERT_CHECK_DERIVATIVES(PAR, Eta)                 \
    ASSERT_CHECK_DERIVATIVES(PAR, Theta)               \
    ASSERT_CHECK_DERIVATIVES(PAR, Phi)                 \
    ASSERT_CHECK_DERIVATIVES(PAR, Et)                  \
    ASSERT_CHECK_DERIVATIVES(PAR, E)                   \
    ASSERT_CHECK_DERIVATIVES(PAR, P)                   \
    ASSERT_CHECK_DERIVATIVES(PAR, Pt)                  \
    ASSERT_CHECK_DERIVATIVES(PAR, PInv)                \
    ASSERT_CHECK_DERIVATIVES(PAR, Px)                  \
    ASSERT_CHECK_DERIVATIVES(PAR, Py)                  \
    ASSERT_CHECK_DERIVATIVES(PAR, Pz)                  \
    ASSERT_CHECK_DERIVATIVES(PAR, M)                   \
  }

IMPL_testDependentVars(Cart) IMPL_testDependentVars(ECart) IMPL_testDependentVars(Spher) IMPL_testDependentVars(ESpher)
    IMPL_testDependentVars(MCCart) IMPL_testDependentVars(MCSpher) IMPL_testDependentVars(MCPInvSpher)
        IMPL_testDependentVars(EtEtaPhi) IMPL_testDependentVars(EtThetaPhi)
    //IMPL_testDependentVars(MomDev)
    //IMPL_testDependentVars(EMomDev)
    //IMPL_testDependentVars(MCMomDev)
    //IMPL_testDependentVars(EScaledMomDev)
    void testKinResolutions::testDependentVars_MomDev() {}
void testKinResolutions::testDependentVars_EMomDev() {}
void testKinResolutions::testDependentVars_MCMomDev() {}
void testKinResolutions::testDependentVars_EScaledMomDev() {}
