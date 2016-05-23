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
    return std::fabs(iLHS)!=0 ? (std::fabs(iLHS-iRHS)< 1E-6*std::fabs(iLHS)) : (std::fabs(iLHS) == std::fabs(iRHS));
  }
}


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

    BinaryOperatorEvaluator<std::minus<double>> be( std::move(cl), std::move(cr), EvaluatorBase::Precedence::kPlusMinus);

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
    reco::FormulaEvaluator f("3<=2");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 0. );
  }
  {
    reco::FormulaEvaluator f("2<=3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 1. );
  }
  {
    reco::FormulaEvaluator f("3<=3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 1. );
  }

  {
    reco::FormulaEvaluator f("3>=2");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 1. );
  }
  {
    reco::FormulaEvaluator f("2>=3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 0. );
  }
  {
    reco::FormulaEvaluator f("3>=3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 1. );
  }


  {
    reco::FormulaEvaluator f("3>2");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 1. );
  }
  {
    reco::FormulaEvaluator f("2>3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 0. );
  }
  {
    reco::FormulaEvaluator f("3>3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 0. );
  }

  {
    reco::FormulaEvaluator f("3<2");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 0. );
  }
  {
    reco::FormulaEvaluator f("2<3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 1. );
  }
  {
    reco::FormulaEvaluator f("3<3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 0. );
  }

  {
    reco::FormulaEvaluator f("2==3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 0. );
  }
  {
    reco::FormulaEvaluator f("3==3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 1. );
  }

  {
    reco::FormulaEvaluator f("2!=3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 1. );
  }
  {
    reco::FormulaEvaluator f("3!=3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 0. );
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
    reco::FormulaEvaluator f("4/2*3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 6. );
  }


  {
    reco::FormulaEvaluator f("1-2+3");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 2. );
  }

  {
    reco::FormulaEvaluator f("(1+2)-(3+4)");
    std::vector<double> emptyV;
    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == -4. );
  }


  {
    reco::FormulaEvaluator f("3/2*4+1");

    std::vector<double> emptyV;
    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 3./2.*4.+1 );
  }

  {
    reco::FormulaEvaluator f("1+3/2*4");
    std::vector<double> emptyV;
    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 1+3./2.*4. );
  }

  {
    reco::FormulaEvaluator f("1+4*(3/2+5)");
    std::vector<double> emptyV;
    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 1+4*(3./2.+5.) );
  }

  {
    reco::FormulaEvaluator f("1+2*3/4*5");
    std::vector<double> emptyV;
    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 1+2.*3./4.*5 );
  }
      
  {
    reco::FormulaEvaluator f("1+2*3/(4+5)+6");
    std::vector<double> emptyV;
    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 1+2.*3./(4+5)+6 );
  }


  {
    reco::FormulaEvaluator f("100./3.*2+1");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 100./3.*2.+1 );
  }

  {
    reco::FormulaEvaluator f("100./3.*(4-2)+2*(3+1)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 100./3.*(4-2)+2*(3+1) );
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
    reco::FormulaEvaluator f("[0]+[1]*3");
    
    std::vector<double> emptyV;
    std::array<double,2> v = {{1.,3.}};

    CPPUNIT_ASSERT( f.evaluate(emptyV,v) == 10. );
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
    reco::FormulaEvaluator f("pow(2,0.3)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == std::pow(2.,0.3) );
  }

  {
    reco::FormulaEvaluator f("TMath::Power(2,0.3)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == std::pow(2.,0.3) );
  }

  {
    reco::FormulaEvaluator f("TMath::Erf(2.)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == TMath::Erf(2.) );
  }

  {
    reco::FormulaEvaluator f("erf(2.)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == std::erf(2.) );
  }

  {
    reco::FormulaEvaluator f("TMath::Landau(3.)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == TMath::Landau(3.) );
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
    reco::FormulaEvaluator f("TMath::Max(2,1)");
    
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == 2 );
  }

  {
    reco::FormulaEvaluator f("TMath::Max(1,2)");
    
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

  {
    reco::FormulaEvaluator f("-(-2.36997+0.413917*TMath::Log(208))/208");
    std::vector<double> emptyV;

    CPPUNIT_ASSERT( f.evaluate(emptyV,emptyV) == -(-2.36997+0.413917*std::log(208.))/208.);
  }

  {
    //For Jet energy corrections
    reco::FormulaEvaluator f("2*TMath::Erf(4*(x-1))");

    std::vector<double> x ={1.};


    std::vector<double> xValues = {1., 2., 3.};
    std::vector<double> emptyV;

    auto func = [](double x) { return 2*TMath::Erf(4*(x-1)); };

    for(auto const xv: xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, emptyV),func(x[0])) );
    }
  }

  {
    //For Jet energy corrections
    reco::FormulaEvaluator f("2*TMath::Landau(2*(x-1))");

    std::vector<double> x ={1.};


    std::vector<double> xValues = {1., 2., 3.};
    std::vector<double> emptyV;

    auto func = [](double x) { return 2*TMath::Landau(2*(x-1)); };

    for(auto const xv: xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, emptyV),func(x[0])) );
    }
  }

  {
    //From SimpleJetCorrector
    reco::FormulaEvaluator f("([0]+([1]/((log10(x)^2)+[2])))+([3]*exp(-([4]*((log10(x)-[5])*(log10(x)-[5])))))");

    std::vector<double> x ={1.};

    std::vector<double> v = {1.,4.,2.,0.5,2.,1.};

    std::vector<double> xValues = {1., 10., 100.};

    auto func = [&v](double x) { return (v[0]+(v[1]/(( (std::log(x)/std::log(10))*(std::log(x)/std::log(10)) ) +v[2])))+(v[3]*std::exp(-1.*(v[4]*((std::log(x)/std::log(10.)-v[5])*(std::log(x)/std::log(10.)-v[5]))))); };

    for(auto const xv: xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v),func(x[0])) );
    }
  }

  {
    //From SimpleJetCorrector
    reco::FormulaEvaluator f("[0]*([1]+[2]*TMath::Log(x))");

    std::vector<double> x ={1.};

    std::vector<double> v = {1.3,4.,2.};

    std::vector<double> xValues = {1., 10., 100.};

    auto func = [&v](double x) { return v[0]*(v[1]+v[2]*std::log(x)); };

    for(auto const xv: xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v),func(x[0])) );
    }
  }

  {
    //From SimpleJetCorrector
    reco::FormulaEvaluator f("[0]+([1]/((log10(x)^[2])+[3]))");

    std::vector<double> x ={1.};

    std::vector<double> v = {1.3,4.,1.7,1.};

    std::vector<double> xValues = {1., 10., 100.};

    auto func = [&v](double x) { return v[0]+(v[1]/(( std::pow(log(x)/log(10.),v[2]) )+v[3])); };

    for(auto const xv: xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v),func(x[0])) );
    }
  }

  {
    //From SimpleJetCorrector
    reco::FormulaEvaluator f("max(0.0001,1-y*([0]+([1]*z)*(1+[2]*log(x)))/x)");

    std::vector<double> v ={.1,1.,.5};

    std::vector<double> p = {1.3,5.,10.};

    std::vector<double> xValues = {1., 10., 100.};

    auto func = [&p](double x, double y, double z) { return std::max(0.0001, 1-y*(p[0]+(p[1]*z)*(1+p[2]*std::log(x)))/x); };

    for(auto const xv: xValues) {
      v[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(v, p),func(v[0],v[1],v[2])) );
    }
  }
  {
    reco::FormulaEvaluator f("(-2.36997+0.413917*TMath::Log(x))/x-(-2.36997+0.413917*TMath::Log(208))/208");

    std::vector<double> x ={1.};

    std::vector<double> v;

    auto func = [](double x) {return (-2.36997+0.413917*std::log(x))/x-(-2.36997+0.413917*std::log(208))/208;};

    std::vector<double> xValues = {.1, 1., 10., 100.};
    for(auto const xv: xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v),func(x[0])) );
    }
  }

  {
    reco::FormulaEvaluator f("TMath::Max(0.,1.03091-0.051154*pow(x,-0.154227))-TMath::Max(0.,1.03091-0.051154*TMath::Power(208.,-0.154227))");
    std::vector<double> x ={1.};

    std::vector<double> v;


    std::vector<double> xValues = {.1, 1., 10., 100.};


    auto func = [](double x) { return std::max(0.,1.03091-0.051154*std::pow(x,-0.154227))-std::max(0.,1.03091-0.051154*std::pow(208.,-0.154227)); };

    for(auto const xv: xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v),func(x[0])) );
    }
  }

  {
    reco::FormulaEvaluator f("[2]*([3]+[4]*TMath::Log(max([0],min([1],x))))");

    std::vector<double> x = {1.};

    std::vector<double> v = {1.,4.,2.,0.5,2.,1.,1., -1.};
    std::vector<double> xValues = {.1, 1., 10., 100.};

    auto func =[&v](double x) { return v[2]*(v[3]+v[4]*std::log(std::max(v[0],std::min(v[1],x)))); };
    for(auto const xv: xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v),func(x[0])) );
    }
  }
  {
    //From SimpleJetCorrector
    reco::FormulaEvaluator f("((x>=[6])*(([0]+([1]/((log10(x)^2)+[2])))+([3]*exp(-([4]*((log10(x)-[5])*(log10(x)-[5])))))))+((x<[6])*[7])");

    std::vector<double> x ={1.};

    std::vector<double> v = {1.,4.,2.,0.5,2.,1.,1., -1.};


    std::vector<double> xValues = {.1, 1., 10., 100.};

    auto func = [&v](double x) { return ((x>=v[6])*((v[0]+(v[1]/(( (std::log(x)/std::log(10))*(std::log(x)/std::log(10)) ) +v[2])))+(v[3]*std::exp(-1.*(v[4]*((std::log(x)/std::log(10.)-v[5])*(std::log(x)/std::log(10.)-v[5])))))))+((x<v[6])*v[7]); };


    for(auto const xv: xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v),func(x[0])) );
    }
  }


  {
    reco::FormulaEvaluator f("(TMath::Max(0.,1.03091-0.051154*pow(x,-0.154227))-TMath::Max(0.,1.03091-0.051154*TMath::Power(208.,-0.154227)))+[7]*((-2.36997+0.413917*TMath::Log(x))/x-(-2.36997+0.413917*TMath::Log(208))/208)");

    std::vector<double> x = {1.};

    std::vector<double> v = {1.,4.,2.,0.5,2.,1.,1., -1.};
    std::vector<double> xValues = {.1, 1., 10., 100.};


    auto func =[&v](double x) { return (std::max(0.,1.03091-0.051154*std::pow(x,-0.154227))-std::max(0.,1.03091-0.051154*std::pow(208.,-0.154227)))+v[7]*((-2.36997+0.413917*std::log(x))/x-(-2.36997+0.413917*std::log(208))/208); };

    for(auto const xv: xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v),func(x[0])) );
    }
  }

  {
    //From SimpleJetCorrector
    reco::FormulaEvaluator f("100./3.*0.154227+2.36997");

    std::vector<double> x = {1.};

    std::vector<double> v = {1.,4.,2.,0.5,2.,1.,1., -1.};
    std::vector<double> xValues = {.1, 1., 10., 100.};


    auto func =[&v](double x) {return 100./3.*0.154227+2.36997;
    };

    for(auto const xv: xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v),func(x[0])) );
    }
  }

  {
    //From SimpleJetCorrector
    reco::FormulaEvaluator f("[2]*([3]+[4]*TMath::Log(max([0],min([1],x))))*1./([5]+[6]*100./3.*(TMath::Max(0.,1.03091-0.051154*pow(x,-0.154227))-TMath::Max(0.,1.03091-0.051154*TMath::Power(208.,-0.154227)))+[7]*((-2.36997+0.413917*TMath::Log(x))/x-(-2.36997+0.413917*TMath::Log(208))/208))");

    std::vector<double> x = {1.};

    std::vector<double> v = {1.,4.,2.,0.5,2.,1.,1., -1.};
    std::vector<double> xValues = {.1, 1., 10., 100.};


    auto func =[&v](double x) {return v[2]*(v[3]+v[4]*std::log(std::max(v[0],std::min(v[1],x))))*1./(v[5]+v[6]*100./3.*(std::max(0.,1.03091-0.051154*std::pow(x,-0.154227))-std::max(0.,1.03091-0.051154*std::pow(208.,-0.154227)))+v[7]*((-2.36997+0.413917*std::log(x))/x-(-2.36997+0.413917*std::log(208))/208));
    };

    for(auto const xv: xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v),func(x[0])) );
    }
  }

  {
    //Tests that pick proper evaluator for argument of function
    reco::FormulaEvaluator f("exp([4]*(log10(x)-[5])*(log10(x)-[5]))");
    std::vector<double> x = {10.};

    std::vector<double> v = {0.88524, 28.4947, 4.89135, -19.0245, 0.0227809, -6.97308};
    std::vector<double> xValues = {10.};

    auto func =[&v](double x) {return std::exp(v[4]*(std::log(x)/std::log(10)-v[5])*(std::log(x)/std::log(10)-v[5])); }; 

    for(auto const xv: xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

  {
    reco::FormulaEvaluator f("max(0.0001,[0]+[1]/(pow(log10(x),2)+[2])+[3]*exp(-1*([4]*((log10(x)-[5])*(log10(x)-[5])))))");

    std::vector<double> x = {10.};

    std::vector<double> v = {0.88524, 28.4947, 4.89135, -19.0245, 0.0227809, -6.97308};
    std::vector<double> xValues = {10.};


    auto func =[&v](double x) {return std::max(0.0001,v[0]+v[1]/(std::pow(std::log(x)/std::log(10), 2)+v[2])+v[3]*std::exp(-1*v[4]*(std::log(x)/std::log(10)-v[5])*(std::log(x)/std::log(10)-v[5]))); };

    for(auto const xv: xValues) {
      x[0] = xv;
      CPPUNIT_ASSERT(compare(f.evaluate(x, v), func(x[0])));
    }
  }

}
