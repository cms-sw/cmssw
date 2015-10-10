#include <cppunit/extensions/HelperMacros.h>
#include "CommonTools/Utils/src/formulaConstantEvaluator.h"
#include "CommonTools/Utils/src/formulaParameterEvaluator.h"
#include "CommonTools/Utils/src/formulaVariableEvaluator.h"
#include "CommonTools/Utils/src/formulaBinaryOperatorEvaluator.h"
#include "CommonTools/Utils/src/formulaFunctionOneArgEvaluator.h"
#include "CommonTools/Utils/src/formulaFunctionTwoArgsEvaluator.h"
#include "CommonTools/Utils/src/formulaUnaryMinusEvaluator.h"

#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include <algorithm>

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

CPPUNIT_TEST_SUITE_REGISTRATION(testFormulaEvaluator);

void testFormulaEvaluator::checkEvaluators() {
  using namespace reco::formula;
  {
    ConstantEvaluator c{4};

    CPPUNIT_ASSERT( c.evaluate(nullptr,nullptr) == 4. );
  }

  {
    ParameterEvaluator pe{0};

    double p = 5.;

    CPPUNIT_ASSERT( pe.evaluate(nullptr, &p) == p);
  }

  {
    VariableEvaluator ve{0};

    double v = 3.;

    CPPUNIT_ASSERT( ve.evaluate(&v, nullptr) == v);
  }

  {
    auto cl = std::unique_ptr<ConstantEvaluator>( new ConstantEvaluator(4) );
    auto cr = std::unique_ptr<ConstantEvaluator>( new ConstantEvaluator(3) );

    BinaryOperatorEvaluator<std::minus<double>> be( std::move(cl), std::move(cr), EvaluatorBase::Precidence::kPlusMinus);

    CPPUNIT_ASSERT( be.evaluate(nullptr,nullptr) == 1. );
  }

  {
    auto cl = std::unique_ptr<ConstantEvaluator>( new ConstantEvaluator(4) );

    FunctionOneArgEvaluator f( std::move(cl), [](double v) { return std::exp(v); } );

    CPPUNIT_ASSERT( f.evaluate(nullptr,nullptr) == std::exp(4.) );
  }

  {
    auto cl = std::unique_ptr<ConstantEvaluator>( new ConstantEvaluator(4) );
    auto cr = std::unique_ptr<ConstantEvaluator>( new ConstantEvaluator(3) );

    FunctionTwoArgsEvaluator f( std::move(cl), std::move(cr),
                                [](double v1, double v2) { return std::max(v1,v2); });

    CPPUNIT_ASSERT( f.evaluate(nullptr,nullptr) == 4. );
  }

  {
    auto cl = std::unique_ptr<ConstantEvaluator>( new ConstantEvaluator(4) );

    UnaryMinusEvaluator f( std::move(cl) );

    CPPUNIT_ASSERT( f.evaluate(nullptr,nullptr) == -4. );
  }

}

void 
testFormulaEvaluator::checkFormulaEvaluator() {
  {
    reco::FormulaEvaluator f("5");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 5. );
  }

  {
    reco::FormulaEvaluator f("3+2");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 5. );
  }

  {
    reco::FormulaEvaluator f("3-2");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 1. );
  }

  {
    reco::FormulaEvaluator f("3*2");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 6. );
  }

  {
    reco::FormulaEvaluator f("6/2");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 3. );
  }

  {
    reco::FormulaEvaluator f("3^2");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 9. );
  }


  {
    reco::FormulaEvaluator f("1+2*3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 7. );
  }

  {
    reco::FormulaEvaluator f("(1+2)*3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 9. );
  }

  {
    reco::FormulaEvaluator f("2*3+1");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 7. );
  }

  {
    reco::FormulaEvaluator f("2*(3+1)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 8. );
  }

  {
    reco::FormulaEvaluator f("x");
    
    std::vector<double> emptyV;
    std::array<double,1> v ={{3.}};

    CPPUNIT_ASSERT( f.evaluate(v,emptyV) == 3. );
  }

  {
    reco::FormulaEvaluator f("y");
    
    std::vector<double> emptyV;
    std::array<double,2> v = {{0.,3.}};

    CPPUNIT_ASSERT( f.evaluate(v,emptyV) == 3. );
  }

  {
    reco::FormulaEvaluator f("z");
    
    std::vector<double> emptyV;
    std::array<double,3> v = {{0.,0.,3.}};

    CPPUNIT_ASSERT( f.evaluate(v,emptyV) == 3. );
  }

  {
    reco::FormulaEvaluator f("t");
    
    std::vector<double> emptyV;
    std::array<double,4> v = {{0.,0.,0.,3.}};

    CPPUNIT_ASSERT( f.evaluate(v,emptyV) == 3. );
  }


  {
    reco::FormulaEvaluator f("[0]");
    
    std::vector<double> emptyV;
    std::array<double,1> v = {{3.}};

    CPPUNIT_ASSERT( f.evaluate(emptyV,v) == 3. );
  }

  {
    reco::FormulaEvaluator f("[1]");
    
    std::vector<double> emptyV;
    std::array<double,2> v = {{0.,3.}};

    CPPUNIT_ASSERT( f.evaluate(emptyV,v) == 3. );
  }

  {
    reco::FormulaEvaluator f("log(2)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == std::log(2.) );
  }
  {
    reco::FormulaEvaluator f("TMath::Log(2)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == std::log(2.) );
  }

  {
    reco::FormulaEvaluator f("log10(2)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == std::log(2.)/std::log(10.) );
  }

  {
    reco::FormulaEvaluator f("exp(2)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == std::exp(2.) );
  }

  {
    reco::FormulaEvaluator f("max(2,1)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 2 );
  }

  {
    reco::FormulaEvaluator f("max(1,2)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 2 );
  }

  {
    reco::FormulaEvaluator f("max(max(5,3),2)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 5 );
  }

  {
    reco::FormulaEvaluator f("max(2,max(5,3))");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 5 );
  }

}
