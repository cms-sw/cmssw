#include <cppunit/extensions/HelperMacros.h>
#include "CommonTools/Utils/src/formulaConstantEvaluator.h"
#include "CommonTools/Utils/src/formulaParameterEvaluator.h"
#include "CommonTools/Utils/src/formulaVariableEvaluator.h"
#include "CommonTools/Utils/src/formulaBinaryOperatorEvaluator.h"
#include "CommonTools/Utils/src/formulaFunctionOneArgEvaluator.h"
#include "CommonTools/Utils/src/formulaFunctionTwoArgsEvaluator.h"
#include "CommonTools/Utils/src/formulaUnaryMinusEvaluator.h"

#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <cmath>
#include "TMath.h"

class testFormulaEvaluator : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testFormulaEvaluator);
  CPPUNIT_TEST(checkEvaluators);
  CPPUNIT_TEST(checkFormulaEvaluator);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkEvaluators();
  void checkFormulaEvaluator();
};

namespace {
  bool compare(double iLHS, double iRHS) {
    return std::fabs(iLHS) != 0 ? (std::fabs(iLHS - iRHS) < 1E-6 * std::fabs(iLHS))
                                : (std::fabs(iLHS) == std::fabs(iRHS));
  }
}  // namespace

CPPUNIT_TEST_SUITE_REGISTRATION(testFormulaEvaluator);

void testFormulaEvaluator::checkEvaluators() {
  using namespace reco::formula;
  {
    ConstantEvaluator c{4};

    CPPUNIT_ASSERT(c.evaluate(nullptr, nullptr) == 4.);
  }

  {
    ParameterEvaluator pe{0};

    double p = 5.;

    CPPUNIT_ASSERT(pe.evaluate(nullptr, &p) == p);
  }

  {
    VariableEvaluator ve{0};

    double v = 3.;

    CPPUNIT_ASSERT(ve.evaluate(&v, nullptr) == v);
  }

  {
    auto cl = std::unique_ptr<ConstantEvaluator>(new ConstantEvaluator(4));
    auto cr = std::unique_ptr<ConstantEvaluator>(new ConstantEvaluator(3));

    BinaryOperatorEvaluator<std::minus<double>> be(std::move(cl), std::move(cr), EvaluatorBase::Precedence::kPlusMinus);

    CPPUNIT_ASSERT(be.evaluate(nullptr, nullptr) == 1.);
  }

  {
    auto cl = std::unique_ptr<ConstantEvaluator>(new ConstantEvaluator(4));

    FunctionOneArgEvaluator f(std::move(cl), [](double v) { return std::exp(v); });

    CPPUNIT_ASSERT(f.evaluate(nullptr, nullptr) == std::exp(4.));
  }

  {
    auto cl = std::unique_ptr<ConstantEvaluator>(new ConstantEvaluator(4));
    auto cr = std::unique_ptr<ConstantEvaluator>(new ConstantEvaluator(3));

    FunctionTwoArgsEvaluator f(std::move(cl), std::move(cr), [](double v1, double v2) { return std::max(v1, v2); });

    CPPUNIT_ASSERT(f.evaluate(nullptr, nullptr) == 4.);
  }

  {
    auto cl = std::unique_ptr<ConstantEvaluator>(new ConstantEvaluator(4));

    UnaryMinusEvaluator f(std::move(cl));

    CPPUNIT_ASSERT(f.evaluate(nullptr, nullptr) == -4.);
  }
}

void testFormulaEvaluator::checkFormulaEvaluator() {
  {
    reco::FormulaEvaluator f("5");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 5.);
  }

  {
    reco::FormulaEvaluator f("3+2");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 5.);
  }

  {
    reco::FormulaEvaluator f(" 3 + 2 ");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 5.);
  }

  {
    reco::FormulaEvaluator f("3-2");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1.);
  }

  {
    reco::FormulaEvaluator f("3*2");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 6.);
  }

  {
    reco::FormulaEvaluator f("6/2");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 3.);
  }

  {
    reco::FormulaEvaluator f("3^2");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 9.);
  }

  {
    reco::FormulaEvaluator f("4*3^2");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 36.);
  }
  {
    reco::FormulaEvaluator f("3^2*4");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 36.);
  }

  {
    reco::FormulaEvaluator f("1+2*3^4+5*2+6*2");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1 + 2 * (3 * 3 * 3 * 3) + 5 * 2 + 6 * 2);
  }

  {
    reco::FormulaEvaluator f("1+3^4*2+5*2+6*2");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1 + 2 * (3 * 3 * 3 * 3) + 5 * 2 + 6 * 2);
  }

  {
    reco::FormulaEvaluator f("3<=2");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 0.);
  }
  {
    reco::FormulaEvaluator f("2<=3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1.);
  }
  {
    reco::FormulaEvaluator f("3<=3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1.);
  }

  {
    reco::FormulaEvaluator f("3>=2");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1.);
  }
  {
    reco::FormulaEvaluator f("2>=3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 0.);
  }
  {
    reco::FormulaEvaluator f("3>=3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1.);
  }

  {
    reco::FormulaEvaluator f("3>2");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1.);
  }
  {
    reco::FormulaEvaluator f("2>3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 0.);
  }
  {
    reco::FormulaEvaluator f("3>3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 0.);
  }

  {
    reco::FormulaEvaluator f("3<2");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 0.);
  }
  {
    reco::FormulaEvaluator f("2<3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1.);
  }
  {
    reco::FormulaEvaluator f("3<3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 0.);
  }

  {
    reco::FormulaEvaluator f("2==3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 0.);
  }
  {
    reco::FormulaEvaluator f("3==3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1.);
  }

  {
    reco::FormulaEvaluator f("2!=3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1.);
  }
  {
    reco::FormulaEvaluator f("3!=3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 0.);
  }

  {
    reco::FormulaEvaluator f("1+2*3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 7.);
  }

  {
    reco::FormulaEvaluator f("(1+2)*3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 9.);
  }

  {
    reco::FormulaEvaluator f("2*3+1");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 7.);
  }

  {
    reco::FormulaEvaluator f("2*(3+1)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 8.);
  }

  {
    reco::FormulaEvaluator f("4/2*3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 6.);
  }

  {
    reco::FormulaEvaluator f("1-2+3");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 2.);
  }

  {
    reco::FormulaEvaluator f("(1+2)-(3+4)");
    std::vector<double> emptyV;
    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == -4.);
  }

  {
    reco::FormulaEvaluator f("3/2*4+1");

    std::vector<double> emptyV;
    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 3. / 2. * 4. + 1);
  }

  {
    reco::FormulaEvaluator f("1+3/2*4");
    std::vector<double> emptyV;
    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1 + 3. / 2. * 4.);
  }

  {
    reco::FormulaEvaluator f("1+4*(3/2+5)");
    std::vector<double> emptyV;
    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1 + 4 * (3. / 2. + 5.));
  }

  {
    reco::FormulaEvaluator f("1+2*3/4*5");
    std::vector<double> emptyV;
    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1 + 2. * 3. / 4. * 5);
  }

  {
    reco::FormulaEvaluator f("1+2*3/(4+5)+6");
    std::vector<double> emptyV;
    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 1 + 2. * 3. / (4 + 5) + 6);
  }

  {
    reco::FormulaEvaluator f("100./3.*2+1");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 100. / 3. * 2. + 1);
  }

  {
    reco::FormulaEvaluator f("100./3.*(4-2)+2*(3+1)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 100. / 3. * (4 - 2) + 2 * (3 + 1));
  }

  {
    reco::FormulaEvaluator f("2*(3*4*5)/6");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 2 * (3 * 4 * 5) / 6);
  }

  {
    reco::FormulaEvaluator f("2*(2.5*3*3.5)*(4*4.5*5)/6");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 2 * (2.5 * 3 * 3.5) * (4 * 4.5 * 5) / 6);
  }

  {
    reco::FormulaEvaluator f("x");

    std::vector<double> emptyV;
    std::array<double, 1> v = {{3.}};

    CPPUNIT_ASSERT(f.evaluate(v, emptyV) == 3.);
  }

  {
    reco::FormulaEvaluator f("y");

    std::vector<double> emptyV;
    std::array<double, 2> v = {{0., 3.}};

    CPPUNIT_ASSERT(f.evaluate(v, emptyV) == 3.);
  }

  {
    reco::FormulaEvaluator f("z");

    std::vector<double> emptyV;
    std::array<double, 3> v = {{0., 0., 3.}};

    CPPUNIT_ASSERT(f.evaluate(v, emptyV) == 3.);
  }

  {
    reco::FormulaEvaluator f("t");

    std::vector<double> emptyV;
    std::array<double, 4> v = {{0., 0., 0., 3.}};

    CPPUNIT_ASSERT(f.evaluate(v, emptyV) == 3.);
  }

  {
    reco::FormulaEvaluator f("[0]");

    std::vector<double> emptyV;
    std::array<double, 1> v = {{3.}};

    CPPUNIT_ASSERT(f.evaluate(emptyV, v) == 3.);
  }

  {
    reco::FormulaEvaluator f("[1]");

    std::vector<double> emptyV;
    std::array<double, 2> v = {{0., 3.}};

    CPPUNIT_ASSERT(f.evaluate(emptyV, v) == 3.);
  }

  {
    reco::FormulaEvaluator f("[0]+[1]*3");

    std::vector<double> emptyV;
    std::array<double, 2> v = {{1., 3.}};

    CPPUNIT_ASSERT(f.evaluate(emptyV, v) == 10.);
  }

  {
    reco::FormulaEvaluator f("log(2)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::log(2.));
  }
  {
    reco::FormulaEvaluator f("TMath::Log(2)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::log(2.));
  }

  {
    reco::FormulaEvaluator f("log10(2)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::log(2.) / std::log(10.));
  }

  {
    reco::FormulaEvaluator f("exp(2)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::exp(2.));
  }

  {
    reco::FormulaEvaluator f("pow(2,0.3)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::pow(2., 0.3));
  }

  {
    reco::FormulaEvaluator f("TMath::Power(2,0.3)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::pow(2., 0.3));
  }

  {
    reco::FormulaEvaluator f("TMath::Erf(2.)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::Erf(2.));
  }

  {
    reco::FormulaEvaluator f("erf(2.)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::erf(2.));
  }

  {
    reco::FormulaEvaluator f("TMath::Landau(3.)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::Landau(3.));
  }

  {
    reco::FormulaEvaluator f("max(2,1)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 2);
  }

  {
    reco::FormulaEvaluator f("max(1,2)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 2);
  }

  {
    reco::FormulaEvaluator f("TMath::Max(2,1)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 2);
  }

  {
    reco::FormulaEvaluator f("TMath::Max(1,2)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 2);
  }

  {
    reco::FormulaEvaluator f("cos(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::cos(0.5));
  }

  {
    reco::FormulaEvaluator f("TMath::Cos(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::Cos(0.5));
  }

  {
    reco::FormulaEvaluator f("sin(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::sin(0.5));
  }

  {
    reco::FormulaEvaluator f("TMath::Sin(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::Sin(0.5));
  }

  {
    reco::FormulaEvaluator f("tan(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::tan(0.5));
  }

  {
    reco::FormulaEvaluator f("TMath::Tan(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::Tan(0.5));
  }

  {
    reco::FormulaEvaluator f("acos(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::acos(0.5));
  }

  {
    reco::FormulaEvaluator f("TMath::ACos(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::ACos(0.5));
  }

  {
    reco::FormulaEvaluator f("asin(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::asin(0.5));
  }

  {
    reco::FormulaEvaluator f("TMath::ASin(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::ASin(0.5));
  }

  {
    reco::FormulaEvaluator f("atan(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::atan(0.5));
  }

  {
    reco::FormulaEvaluator f("TMath::ATan(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::ATan(0.5));
  }

  {
    reco::FormulaEvaluator f("atan2(-0.5, 0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::atan2(-0.5, 0.5));
  }

  {
    reco::FormulaEvaluator f("TMath::ATan2(-0.5, 0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::ATan2(-0.5, 0.5));
  }

  {
    reco::FormulaEvaluator f("cosh(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::cosh(0.5));
  }

  {
    reco::FormulaEvaluator f("TMath::CosH(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::CosH(0.5));
  }

  {
    reco::FormulaEvaluator f("sinh(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::sinh(0.5));
  }

  {
    reco::FormulaEvaluator f("TMath::SinH(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::SinH(0.5));
  }

  {
    reco::FormulaEvaluator f("tanh(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::tanh(0.5));
  }

  {
    reco::FormulaEvaluator f("TMath::TanH(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::TanH(0.5));
  }

  // std::acosh and std::atanh are using delta fabs instead of equality because gcc compute the value differently at compile time.
  {
    reco::FormulaEvaluator f("acosh(2.0)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(fabs(f.evaluate(emptyV, emptyV) - std::acosh(2.0)) < 1e-9);
  }

  {
    reco::FormulaEvaluator f("TMath::ACosH(2.0)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::ACosH(2.0));
  }

  {
    reco::FormulaEvaluator f("asinh(2.0)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == std::asinh(2.0));
  }

  {
    reco::FormulaEvaluator f("TMath::ASinH(2.0)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::ASinH(2.0));
  }

  {
    reco::FormulaEvaluator f("atanh(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(fabs(f.evaluate(emptyV, emptyV) - std::atanh(0.5)) < 1e-9);
  }

  {
    reco::FormulaEvaluator f("TMath::ATanH(0.5)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == TMath::ATanH(0.5));
  }

  {
    reco::FormulaEvaluator f("max(max(5,3),2)");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 5);
  }

  {
    reco::FormulaEvaluator f("max(2,max(5,3))");

    std::vector<double> emptyV;

    CPPUNIT_ASSERT(f.evaluate(emptyV, emptyV) == 5);
  }

  {
    reco::FormulaEvaluator f("-(-2.36997+0.413917*TMath::Log(208))/208");
    std::vector<double> emptyV;

    auto value = f.evaluate(emptyV, emptyV);
    CPPUNIT_ASSERT(std::abs(value - (-(-2.36997 + 0.413917 * std::log(208.)) / 208.)) / value < 5.0E-16);
  }

  {
    //For Jet energy corrections
    reco::FormulaEvaluator f("2*TMath::Erf(4*(x-1))");

    std::vector<double> x = {1.};

    std::vector<double> xValues = {1., 2., 3.};
    std::vector<double> emptyV;

    auto func = [](double x) { return 2 * TMath::Erf(4 * (x - 1)); };

    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, emptyV), func(x[0])));
    }
  }

  {
    //For Jet energy corrections
    reco::FormulaEvaluator f("2*TMath::Landau(2*(x-1))");

    std::vector<double> x = {1.};

    std::vector<double> xValues = {1., 2., 3.};
    std::vector<double> emptyV;

    auto func = [](double x) { return 2 * TMath::Landau(2 * (x - 1)); };

    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, emptyV), func(x[0])));
    }
  }

  {
    //From SimpleJetCorrector
    reco::FormulaEvaluator f("([0]+([1]/((log10(x)^2)+[2])))+([3]*exp(-([4]*((log10(x)-[5])*(log10(x)-[5])))))");

    std::vector<double> x = {1.};

    std::vector<double> v = {1., 4., 2., 0.5, 2., 1.};

    std::vector<double> xValues = {1., 10., 100.};

    auto func = [&v](double x) {
      return (v[0] + (v[1] / (((std::log(x) / std::log(10)) * (std::log(x) / std::log(10))) + v[2]))) +
             (v[3] *
              std::exp(-1. * (v[4] * ((std::log(x) / std::log(10.) - v[5]) * (std::log(x) / std::log(10.) - v[5])))));
    };

    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    //From SimpleJetCorrector
    reco::FormulaEvaluator f("[0]*([1]+[2]*TMath::Log(x))");

    std::vector<double> x = {1.};

    std::vector<double> v = {1.3, 4., 2.};

    std::vector<double> xValues = {1., 10., 100.};

    auto func = [&v](double x) { return v[0] * (v[1] + v[2] * std::log(x)); };

    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    //From SimpleJetCorrector
    reco::FormulaEvaluator f("[0]+([1]/((log10(x)^[2])+[3]))");

    std::vector<double> x = {1.};

    std::vector<double> v = {1.3, 4., 1.7, 1.};

    std::vector<double> xValues = {1., 10., 100.};

    auto func = [&v](double x) { return v[0] + (v[1] / ((std::pow(log(x) / log(10.), v[2])) + v[3])); };

    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    //From SimpleJetCorrector
    reco::FormulaEvaluator f("max(0.0001,1-y*([0]+([1]*z)*(1+[2]*log(x)))/x)");

    std::vector<double> v = {.1, 1., .5};

    std::vector<double> p = {1.3, 5., 10.};

    std::vector<double> xValues = {1., 10., 100.};

    auto func = [&p](double x, double y, double z) {
      return std::max(0.0001, 1 - y * (p[0] + (p[1] * z) * (1 + p[2] * std::log(x))) / x);
    };

    for (auto const xv : xValues) {
      v[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(v, p), func(v[0], v[1], v[2])));
    }
  }
  {
    reco::FormulaEvaluator f("(-2.36997+0.413917*TMath::Log(x))/x-(-2.36997+0.413917*TMath::Log(208))/208");

    std::vector<double> x = {1.};

    std::vector<double> v;

    auto func = [](double x) {
      return (-2.36997 + 0.413917 * std::log(x)) / x - (-2.36997 + 0.413917 * std::log(208)) / 208;
    };

    std::vector<double> xValues = {.1, 1., 10., 100.};
    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    reco::FormulaEvaluator f(
        "TMath::Max(0.,1.03091-0.051154*pow(x,-0.154227))-TMath::Max(0.,1.03091-0.051154*TMath::Power(208.,-0.154227)"
        ")");
    std::vector<double> x = {1.};

    std::vector<double> v;

    std::vector<double> xValues = {.1, 1., 10., 100.};

    auto func = [](double x) {
      return std::max(0., 1.03091 - 0.051154 * std::pow(x, -0.154227)) -
             std::max(0., 1.03091 - 0.051154 * std::pow(208., -0.154227));
    };

    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    reco::FormulaEvaluator f("[2]*([3]+[4]*TMath::Log(max([0],min([1],x))))");

    std::vector<double> x = {1.};

    std::vector<double> v = {1., 4., 2., 0.5, 2., 1., 1., -1.};
    std::vector<double> xValues = {.1, 1., 10., 100.};

    auto func = [&v](double x) { return v[2] * (v[3] + v[4] * std::log(std::max(v[0], std::min(v[1], x)))); };
    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }
  {
    //From SimpleJetCorrector
    reco::FormulaEvaluator f(
        "((x>=[6])*(([0]+([1]/((log10(x)^2)+[2])))+([3]*exp(-([4]*((log10(x)-[5])*(log10(x)-[5])))))))+((x<[6])*[7])");

    std::vector<double> x = {1.};

    std::vector<double> v = {1., 4., 2., 0.5, 2., 1., 1., -1.};

    std::vector<double> xValues = {.1, 1., 10., 100.};

    auto func = [&v](double x) {
      return ((x >= v[6]) * ((v[0] + (v[1] / (((std::log(x) / std::log(10)) * (std::log(x) / std::log(10))) + v[2]))) +
                             (v[3] * std::exp(-1. * (v[4] * ((std::log(x) / std::log(10.) - v[5]) *
                                                             (std::log(x) / std::log(10.) - v[5]))))))) +
             ((x < v[6]) * v[7]);
    };

    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    reco::FormulaEvaluator f(
        "(TMath::Max(0.,1.03091-0.051154*pow(x,-0.154227))-TMath::Max(0.,1.03091-0.051154*TMath::Power(208.,-0.154227))"
        ")+[7]*((-2.36997+0.413917*TMath::Log(x))/x-(-2.36997+0.413917*TMath::Log(208))/208)");

    std::vector<double> x = {1.};

    std::vector<double> v = {1., 4., 2., 0.5, 2., 1., 1., -1.};
    std::vector<double> xValues = {.1, 1., 10., 100.};

    auto func = [&v](double x) {
      return (std::max(0., 1.03091 - 0.051154 * std::pow(x, -0.154227)) -
              std::max(0., 1.03091 - 0.051154 * std::pow(208., -0.154227))) +
             v[7] * ((-2.36997 + 0.413917 * std::log(x)) / x - (-2.36997 + 0.413917 * std::log(208)) / 208);
    };

    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    //From SimpleJetCorrector
    reco::FormulaEvaluator f("100./3.*0.154227+2.36997");

    std::vector<double> x = {1.};

    std::vector<double> v = {1., 4., 2., 0.5, 2., 1., 1., -1.};
    std::vector<double> xValues = {.1, 1., 10., 100.};

    auto func = [](double x) { return 100. / 3. * 0.154227 + 2.36997; };

    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    //From SimpleJetCorrector
    reco::FormulaEvaluator f(
        "[2]*([3]+[4]*TMath::Log(max([0],min([1],x))))*1./([5]+[6]*100./"
        "3.*(TMath::Max(0.,1.03091-0.051154*pow(x,-0.154227))-TMath::Max(0.,1.03091-0.051154*TMath::Power(208.,-0."
        "154227)))+[7]*((-2.36997+0.413917*TMath::Log(x))/x-(-2.36997+0.413917*TMath::Log(208))/208))");

    std::vector<double> x = {1.};

    std::vector<double> v = {1., 4., 2., 0.5, 2., 1., 1., -1.};
    std::vector<double> xValues = {.1, 1., 10., 100.};

    auto func = [&v](double x) {
      return v[2] * (v[3] + v[4] * std::log(std::max(v[0], std::min(v[1], x)))) * 1. /
             (v[5] +
              v[6] * 100. / 3. *
                  (std::max(0., 1.03091 - 0.051154 * std::pow(x, -0.154227)) -
                   std::max(0., 1.03091 - 0.051154 * std::pow(208., -0.154227))) +
              v[7] * ((-2.36997 + 0.413917 * std::log(x)) / x - (-2.36997 + 0.413917 * std::log(208)) / 208));
    };

    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    //Tests that pick proper evaluator for argument of function
    reco::FormulaEvaluator f("exp([4]*(log10(x)-[5])*(log10(x)-[5]))");
    std::vector<double> x = {10.};

    std::vector<double> v = {0.88524, 28.4947, 4.89135, -19.0245, 0.0227809, -6.97308};
    std::vector<double> xValues = {10.};

    auto func = [&v](double x) {
      return std::exp(v[4] * (std::log(x) / std::log(10) - v[5]) * (std::log(x) / std::log(10) - v[5]));
    };

    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    reco::FormulaEvaluator f(
        "max(0.0001,[0]+[1]/(pow(log10(x),2)+[2])+[3]*exp(-1*([4]*((log10(x)-[5])*(log10(x)-[5])))))");

    std::vector<double> x = {10.};

    std::vector<double> v = {0.88524, 28.4947, 4.89135, -19.0245, 0.0227809, -6.97308};
    std::vector<double> xValues = {10.};

    auto func = [&v](double x) {
      return std::max(
          0.0001,
          v[0] + v[1] / (std::pow(std::log(x) / std::log(10), 2) + v[2]) +
              v[3] * std::exp(-1 * v[4] * (std::log(x) / std::log(10) - v[5]) * (std::log(x) / std::log(10) - v[5])));
    };

    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    std::vector<double> x = {425.92155818};
    std::vector<double> v = {0.945459, 2.78658, 1.65054, -48.1061, 0.0287239, -10.8759};
    std::vector<double> xValues = {425.92155818};

    reco::FormulaEvaluator f("-[4]*(log10(x)-[5])*(log10(x)-[5])");
    auto func = [&v](double x) { return -v[4] * (std::log10(x) - v[5]) * (std::log10(x) - v[5]); };

    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    reco::FormulaEvaluator f("max(0.0001,[0]+[1]/(pow(log10(x),2)+[2])+[3]*exp(-[4]*(log10(x)-[5])*(log10(x)-[5])))");

    std::vector<double> x = {10.};

    std::vector<double> v = {0.88524, 28.4947, 4.89135, -19.0245, 0.0227809, -6.97308};
    std::vector<double> xValues = {10.};

    auto func = [&v](double x) {
      return std::max(
          0.0001,
          v[0] + v[1] / (std::pow(std::log(x) / std::log(10), 2) + v[2]) +
              v[3] * std::exp(-v[4] * (std::log(x) / std::log(10) - v[5]) * (std::log(x) / std::log(10) - v[5])));
    };

    for (auto const xv : xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    reco::FormulaEvaluator f(
        "[2]*([3]*([4]+[5]*TMath::Log(max([0],min([1],x))))*1./([6]+[7]*100./"
        "3.*(TMath::Max(0.,1.03091-0.051154*pow(x,-0.154227))-TMath::Max(0.,1.03091-0.051154*TMath::Power(208.,-0."
        "154227)))+[8]*((1+0.04432-1.304*pow(max(30.,min(6500.,x)),-0.4624)+(0+1.724*TMath::Log(max(30.,min(6500.,x))))"
        "/max(30.,min(6500.,x)))-(1+0.04432-1.304*pow(208.,-0.4624)+(0+1.724*TMath::Log(208.))/208.))))");

    std::vector<double> v = {55, 2510, 0.997756, 1.000155, 0.979016, 0.001834, 0.982, -0.048, 1.250};

    std::vector<double> x = {100};

    auto func = [&v](double x) {
      return v[2] *
             (v[3] * (v[4] + v[5] * TMath::Log(std::max(v[0], std::min(v[1], x)))) * 1. /
              (v[6] +
               v[7] * 100. / 3. *
                   (TMath::Max(0., 1.03091 - 0.051154 * std::pow(x, -0.154227)) -
                    TMath::Max(0., 1.03091 - 0.051154 * TMath::Power(208., -0.154227))) +
               v[8] *
                   ((1 + 0.04432 - 1.304 * std::pow(std::max(30., std::min(6500., x)), -0.4624) +
                     (0 + 1.724 * TMath::Log(std::max(30., std::min(6500., x)))) / std::max(30., std::min(6500., x))) -
                    (1 + 0.04432 - 1.304 * std::pow(208., -0.4624) + (0 + 1.724 * TMath::Log(208.)) / 208.))));
    };

    CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
  }

  {
    std::vector<std::pair<std::string, double>> formulas = {
        {"(1+0.04432+(1.724+100.))-1", 101.76832},
        {"(1+(1.724+100.)+0.04432)-1", 101.76832},
        {"((1.724+100.)+1+0.04432)-1", 101.76832},
        {"(1+0.04432+1.724/100.)-1", .06156},
        {"(1+1.724/100.+0.04432)-1", .06156},
        {"(1.724/100.+1+0.04432)-1", .06156},
        {"(1+0.04432+(1.724/100.))-1", (1 + 0.04432 + (1.724 / 100.)) - 1},
        {"(1+(1.724/100.)+0.04432)-1", (1 + 0.04432 + (1.724 / 100.)) - 1},
        {"((1.724/100.)+1+0.04432)-1", (1 + 0.04432 + (1.724 / 100.)) - 1},
        {"0.997756*(1.000155*(0.979016+0.001834*TMath::Log(max(55.,min(2510.,100.))))*1./(0.982+-0.048*100./"
         "3.*(TMath::Max(0.,1.03091-0.051154*pow(100.,-0.154227))-TMath::Max(0.,1.03091-0.051154*TMath::Power(208.,-0."
         "154227)))+1.250*((1+0.04432-1.304*pow(max(30.,min(6500.,100.)),-0.4624)+(0+1.724*TMath::Log(max(30.,min(6500."
         ",100.))))/max(30.,min(6500.,100.)))-(1+0.04432-1.304*pow(208.,-0.4624)+(0+1.724*TMath::Log(208.))/208.))))",
         0.997756 *
             (1.000155 * (0.979016 + 0.001834 * TMath::Log(std::max(55., std::min(2510., 100.)))) * 1. /
              (0.982 +
               -0.048 * 100. / 3. *
                   (TMath::Max(0., 1.03091 - 0.051154 * std::pow(100., -0.154227)) -
                    TMath::Max(0., 1.03091 - 0.051154 * TMath::Power(208., -0.154227))) +
               1.250 * ((1 + 0.04432 - 1.304 * std::pow(std::max(30., std::min(6500., 100.)), -0.4624) +
                         (0 + 1.724 * TMath::Log(std::max(30., std::min(6500., 100.)))) /
                             std::max(30., std::min(6500., 100.))) -
                        (1 + 0.04432 - 1.304 * std::pow(208., -0.4624) + (0 + 1.724 * TMath::Log(208.)) / 208.))))}};

    std::vector<double> x = {};
    std::vector<double> v = {};
    for (auto const& form_val : formulas) {
      reco::FormulaEvaluator f(form_val.first);

      CPPUNIT_ASSERT(compare(f.evaluate(x, v), form_val.second));
    }
  }

  //tests for JER
  {
    reco::FormulaEvaluator f("[0]+[1]*exp(-x/[2])");

    std::vector<double> v = {0.006467, 0.02519, 77.08};
    std::vector<double> x = {100.};

    auto func = [&v](double x) { return v[0] + v[1] * std::exp(-x / v[2]); };

    CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
  }

  {
    reco::FormulaEvaluator f("max(0.0001,1-y/x*([1]*(z-[0])*(1+[2]*log(x/30.))))");

    std::vector<double> v = {1.4, 0.453645, -0.015665};
    std::vector<double> x = {157.2, 0.5, 23.2};

    auto func = [&v](double x, double y, double z) {
      return std::max(0.0001, 1 - y / x * (v[1] * (z - v[0]) * (1 + v[2] * std::log(x / 30.))));
    };

    CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0], x[1], x[2])));
  }
  {
    reco::FormulaEvaluator f("max(0.0001,1-y*[1]*(z-[0])*(1+[2]*log(x/30.))/x)");

    std::vector<double> v = {1.4, 0.453645, -0.015665};
    std::vector<double> x = {157.2, 0.5, 23.2};

    auto func = [&v](double x, double y, double z) {
      return std::max(0.0001, 1 - y / x * (v[1] * (z - v[0]) * (1 + v[2] * std::log(x / 30.))));
    };

    CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0], x[1], x[2])));
  }
  {
    reco::FormulaEvaluator f("max(0.0001,1-y*([1]*(z-[0])*(1+[2]*log(x/30.)))/x)");

    std::vector<double> v = {1.4, 0.453645, -0.015665};
    std::vector<double> x = {157.2, 0.5, 23.2};

    auto func = [&v](double x, double y, double z) {
      return std::max(0.0001, 1 - y * (v[1] * (z - v[0]) * (1 + v[2] * std::log(x / 30.))) / x);
    };

    CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0], x[1], x[2])));
  }

  {
    reco::FormulaEvaluator f("sqrt([0]*abs([0])/(x*x)+[1]*[1]*pow(x,[3])+[2]*[2])");

    std::vector<double> v = {1.326, 0.4209, 0.02223, -0.6704};
    std::vector<double> x = {100.};

    auto func = [&v](double x) {
      return std::sqrt(v[0] * std::abs(v[0]) / (x * x) + v[1] * v[1] * std::pow(x, v[3]) + v[2] * v[2]);
    };

    CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
  }

  {
    reco::FormulaEvaluator f("sqrt([0]*[0]/(x*x)+[1]*[1]/x+[2]*[2])");

    std::vector<double> v = {2.3, 0.20, 0.009};
    std::vector<double> x = {100.};

    auto func = [&v](double x) { return std::sqrt(v[0] * v[0] / (x * x) + v[1] * v[1] / x + v[2] * v[2]); };

    CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
  }

  {
    //This was previously evaluated wrong
    std::array<double, 3> xs = {{224., 225., 226.}};
    std::vector<double> x = {0.};
    std::vector<double> v = {-3., -3., -3., -3., -3., -3.};

    reco::FormulaEvaluator f(
        "([0]+[1]*x+[2]*x^2)*(x<225)+([0]+[1]*225+[2]*225^2+[3]*(x-225)+[4]*(x-225)^2+[5]*(x-225)^3)*(x>225)");

    auto func = [&v](double x) {
      return (v[0] + v[1] * x + v[2] * (x * x)) * (x < 225) +
             (v[0] + v[1] * 225 + v[2] * (225 * 225) + v[3] * (x - 225) + v[4] * ((x - 225) * (x - 225)) +
              v[5] * ((x - 225) * (x - 225) * (x - 225))) *
                 (x > 225);
    };

    for (auto x_i : xs) {
      x[0] = x_i;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    auto t = []() { reco::FormulaEvaluator f("doesNotExist(2)"); };

    CPPUNIT_ASSERT_THROW(t(), cms::Exception);
  }

  {
    auto t = []() { reco::FormulaEvaluator f("doesNotExist(2) + abs(-1)"); };

    CPPUNIT_ASSERT_THROW(t(), cms::Exception);
  }

  {
    auto t = []() { reco::FormulaEvaluator f("abs(-1) + doesNotExist(2)"); };

    CPPUNIT_ASSERT_THROW(t(), cms::Exception);
  }

  {
    auto t = []() { reco::FormulaEvaluator f("abs(-1) + ( 5 * doesNotExist(2))"); };

    CPPUNIT_ASSERT_THROW(t(), cms::Exception);
  }

  {
    auto t = []() { reco::FormulaEvaluator f("( 5 * doesNotExist(2)) + abs(-1)"); };

    CPPUNIT_ASSERT_THROW(t(), cms::Exception);
  }

  {
    auto t = []() { reco::FormulaEvaluator f("TMath::Exp(2)"); };

    CPPUNIT_ASSERT_THROW(t(), cms::Exception);
  }

  {
    //this was previously causing a seg fault
    auto t = []() { reco::FormulaEvaluator f("1 + 2 * 3 + 5 * doesNotExist(2) "); };

    CPPUNIT_ASSERT_THROW(t(), cms::Exception);
  }

  {
    //Make sure spaces are shown in exception message
    try {
      reco::FormulaEvaluator f("1 + 2#");
    } catch (cms::Exception const& e) {
      auto r =
          "An exception of category 'FormulaEvaluatorParseError' occurred.\n"
          "Exception Message:\n"
          "While parsing '1 + 2#' could not parse beyond '1 + 2'\n";
      CPPUNIT_ASSERT(std::string(r) == e.what());
    }
  }
}
