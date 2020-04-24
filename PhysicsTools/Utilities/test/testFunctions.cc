#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/Gaussian.h"
#include "PhysicsTools/Utilities/interface/Exponential.h"
#include "PhysicsTools/Utilities/interface/Identity.h"
#include "PhysicsTools/Utilities/interface/Composition.h"
#include "PhysicsTools/Utilities/interface/Convolution.h"
#include "PhysicsTools/Utilities/interface/Functions.h"
#include "PhysicsTools/Utilities/interface/Variables.h"
#include "PhysicsTools/Utilities/interface/Numerical.h"
#include "PhysicsTools/Utilities/interface/Fraction.h"
#include "PhysicsTools/Utilities/interface/Expression.h"
#include "PhysicsTools/Utilities/interface/FunctClone.h"
#include <iostream>

class testFunctions : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testFunctions);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testFunctions);

struct TestFun {
  TestFun() : gauss_(0, 1) { }
  double operator()(double x) {
    ++ counter_; return gauss_(x);
  }
  void reset() { counter_ = 0; } 
  static size_t counter_;
private:
  funct::Gaussian gauss_;
};

size_t TestFun::counter_ = 0;

void testFunctions::checkAll() {
  using namespace funct;
  {
    Gaussian g1(0, 1);
    Gaussian g2(1, 1);
    Identity i;
    const double epsilon = 1.e-6;
    Sum<Gaussian, Gaussian>::type g1plus2 = g1 + g2;
    Product<Gaussian, Gaussian>::type g1times2 = g1 * g2; 
    Difference<Gaussian, Gaussian>::type g1minus2 = g1 - g2;
    Ratio<Gaussian, Gaussian>::type g1over2 = g1 / g2; 
    Minus<Gaussian>::type gm1 = - g1;
    Composition<Identity, Gaussian>::type gg1 = compose(i, g1);
    double x = 0.5;
    CPPUNIT_ASSERT(std::abs(g1plus2(x) - (g1(x) + g2(x))) < epsilon);
    CPPUNIT_ASSERT(std::abs(g1times2(x) - (g1(x) * g2(x))) < epsilon);
    CPPUNIT_ASSERT(std::abs(g1minus2(x) - (g1(x) - g2(x))) < epsilon);
    CPPUNIT_ASSERT(std::abs(g1over2(x) - (g1(x) / g2(x))) < epsilon);
    CPPUNIT_ASSERT(std::abs(gm1(x) - (-g1(x))) < epsilon);
    Convolution<Gaussian, Gaussian, TrapezoidIntegrator>::type ggt(g1, g1, -5, 5, TrapezoidIntegrator(1000));
    CPPUNIT_ASSERT(std::abs(ggt(0) - g1(0)/sqrt(2.0))<epsilon);
    Convolution<Gaussian, Gaussian, GaussLegendreIntegrator>::type gggl(g1, g1, -5, 5, GaussLegendreIntegrator(1000, epsilon));
    CPPUNIT_ASSERT(std::abs(gggl(0) - g1(0)/sqrt(2.0))<epsilon);
    CPPUNIT_ASSERT(gg1(0) == g1(0));
  }
  {
    double value = 0.123;
    double epsilon = 1.e-8;
    X x;
    x = value;

    Exp<X>::type f_exp = exp(x);
    CPPUNIT_ASSERT(std::abs(f_exp() - exp(value)) < epsilon);
    Sin<X>::type f_sin = sin(x);
    CPPUNIT_ASSERT(std::abs(f_sin() - sin(value)) < epsilon);
    Cos<X>::type f_cos = cos(x);
    CPPUNIT_ASSERT(std::abs(f_cos() - cos(value)) < epsilon);
    Log<X>::type f_log = log(x);
    CPPUNIT_ASSERT(std::abs(f_log() - log(value)) < epsilon);
  }
  {
    Numerical<1> _1;
    CPPUNIT_ASSERT(_1 == 1);
    Numerical<2> _2;
    CPPUNIT_ASSERT(_2 == 2);
    Numerical<3> _3;
    CPPUNIT_ASSERT(_3 == 3);
    CPPUNIT_ASSERT(num<1>() == 1);
    CPPUNIT_ASSERT(num<2>() == 2);
    CPPUNIT_ASSERT(num<3>() == 3);
  }
  {
    Fraction<1,2>::type _1_2;
    CPPUNIT_ASSERT(_1_2 == 0.5);
  }
  {
    X x = 3.141516;
    Expression f = sin(x) * cos(x);
    const double epsilon = 1.e-4;
    CPPUNIT_ASSERT(std::abs(f() - (sin(x) * cos(x))()) < epsilon);
  }
  {
    TestFun f;
    Master<TestFun> g(f);
    Slave<TestFun> g1(g), g2(g);
    const double epsilon = 1.e-5; 
    double y, y1, y2, x;
    CPPUNIT_ASSERT(f.counter_ == 0);
    x = 0.5; y = g(x), y1 = g1(x), y2 = g2(x);
    CPPUNIT_ASSERT(y == y1 && y1 == y2);
    CPPUNIT_ASSERT(f.counter_ == 1);
    CPPUNIT_ASSERT(std::abs(f(x) - y) < epsilon);
    f.reset();
    x = 1.5; y1 = g1(x), y = g(x), y2 = g2(x);
    CPPUNIT_ASSERT(y == y1 && y1 == y2);
    CPPUNIT_ASSERT(f.counter_ == 1);
    CPPUNIT_ASSERT(std::abs(f(x) - y) < epsilon);
    f.reset();
    x = 0.765; y2 = g2(x), y1 = g1(x), y = g(x);
    CPPUNIT_ASSERT(y == y1 && y1 == y2);
    CPPUNIT_ASSERT(f.counter_ == 1); 
    CPPUNIT_ASSERT(std::abs(f(x) - y) < epsilon);
    f.reset();
    g(0.5); 
    CPPUNIT_ASSERT(f.counter_ == 1);
    g(0.5); 
    CPPUNIT_ASSERT(f.counter_ == 2);
    g1(0.5);
    CPPUNIT_ASSERT(f.counter_ == 2);
    g1(0.5);
    CPPUNIT_ASSERT(f.counter_ == 3);
    g(0.5);
    CPPUNIT_ASSERT(f.counter_ == 3);
    f.reset();
    // odd case: slaves don't catch a change in values
    x = 0.123; y = g(x), y1 = g1(0.5), y2 = g2(0.7);
    CPPUNIT_ASSERT(y == y1 && y1 == y2);
    CPPUNIT_ASSERT(f.counter_ == 1);
    f.reset();
    CPPUNIT_ASSERT(std::abs(f(x) - y) < epsilon);
  }
}
