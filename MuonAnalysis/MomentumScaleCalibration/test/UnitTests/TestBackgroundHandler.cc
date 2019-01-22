#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestRunner.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TextTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>

#include <algorithm>
#include <iterator>

#include "MuonAnalysis/MomentumScaleCalibration/interface/BackgroundHandler.h"

#ifndef TestBackgroundHandler_cc
#define TestBackgroundHandler_cc

class TestBackgroundHandler : public CppUnit::TestFixture {
public:
  TestBackgroundHandler() {}
  void setUp()
  {
    fill_n(back_inserter(identifiers), 3, 2);
    // fill_n(back_inserter(leftWindowFactors), 3, 2);
    // fill_n(back_inserter(rightWindowFactors), 3, 2);

    leftWindowBorders.push_back(2.5);
    rightWindowBorders.push_back(3.5);

    leftWindowBorders.push_back(8.1);
    rightWindowBorders.push_back(12.2);

    leftWindowBorders.push_back(80.);
    rightWindowBorders.push_back(100.);

    double tempResMass[] = {91.1876, 10.3552, 10.0233, 9.4603, 3.68609, 3.0969};
    std::copy(tempResMass, tempResMass+6, ResMass);
    double tempMassWindowHalfWidth[] = { 20., 0.5, 0.5, 0.5, 0.2, 0.2 };
    std::copy(tempMassWindowHalfWidth, tempMassWindowHalfWidth+6, massWindowHalfWidth);

    backgroundHandler_ = new BackgroundHandler(identifiers, leftWindowBorders, rightWindowBorders,
                                               ResMass, massWindowHalfWidth);
  }
  void tearDown()
  {
    delete backgroundHandler_;
  }
  void testConstructor()
  {
    CPPUNIT_ASSERT( backgroundHandler_->resonanceWindow_.size() == 6 );

    // Check that the resonance windows contain the correct masses and bounds
    unsigned int i=0;
    for(auto const& resonanceWindow : backgroundHandler_->resonanceWindow_)
    {
      CPPUNIT_ASSERT(resonanceWindow.mass() == ResMass[i]);
      // Convert to float because of precision problems with doubles
      CPPUNIT_ASSERT(float(resonanceWindow.lowerBound()) == float(ResMass[i] - massWindowHalfWidth[i]));
      CPPUNIT_ASSERT(float(resonanceWindow.upperBound()) == float(ResMass[i] + massWindowHalfWidth[i]));
      ++i;
    }

    // Check the background windows
    CPPUNIT_ASSERT( backgroundHandler_->backgroundWindow_.size() == 3 );
    // Check masses and bounds of the background windows
    double resMassForRegion[3];
    resMassForRegion[0] = ResMass[0];
    resMassForRegion[1] = (ResMass[1]+ResMass[2]+ResMass[3])/3;
    resMassForRegion[2] = (ResMass[4]+ResMass[5])/2;
    i = 0;
    for(auto const& backgroundWindow : backgroundHandler_->backgroundWindow_)
    {
      CPPUNIT_ASSERT(backgroundWindow.mass() == resMassForRegion[i]);
      CPPUNIT_ASSERT(float(backgroundWindow.lowerBound()) == float(leftWindowBorders[i]));
      CPPUNIT_ASSERT(float(backgroundWindow.upperBound()) == float(rightWindowBorders[i]));
      ++i;
    }
  }

  void testInitializeParNums()
  {
    CPPUNIT_ASSERT( backgroundHandler_->parNumsRegions_[1] == backgroundHandler_->backgroundWindow_[0].backgroundFunction()->parNum() );
    CPPUNIT_ASSERT( backgroundHandler_->parNumsResonances_[3] == 6*(backgroundHandler_->backgroundWindow_[0].backgroundFunction()->parNum()) );
  }

  void testBackgroundFunction()
  {
    bool resConsidered[6] = {false};
    resConsidered[5] = true;
    // backgroundFunctionBase * bkFun = backgroundHandler_->backgroundWindow_[0].backgroundFunction();
    double parval[] = {0., 0., 0., 0., 0.3, 0.8};
    double mass = 3.;
    std::pair<double, double> result = backgroundHandler_->backgroundFunction( true, parval, 6, 5,
									       resConsidered, ResMass, massWindowHalfWidth,
									       1, mass, 0., 0. );
    double lowerBound = backgroundHandler_->backgroundWindow_[2].lowerBound();
    double upperBound = backgroundHandler_->backgroundWindow_[2].upperBound();
    CPPUNIT_ASSERT( result.first == parval[4] );
    CPPUNIT_ASSERT( float(result.second) == float(-parval[5]*exp(-parval[5]*mass)/(exp(-parval[5]*upperBound) - exp(-parval[5]*lowerBound))) );
  }

  void testSetParameters()
  {
    //double Start
    //backgroundHandler_.setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const std::vector<double> & parBgr, const std::vector<int> & parBgrOrder, const int muonType)

  }

  std::vector<int> identifiers;
  std::vector<double> leftWindowBorders;
  std::vector<double> rightWindowBorders;
  double ResMass[6];
  double massWindowHalfWidth[6];

  BackgroundHandler * backgroundHandler_;

  // Declare and build the test suite
  CPPUNIT_TEST_SUITE( TestBackgroundHandler );
  CPPUNIT_TEST( testConstructor );
  CPPUNIT_TEST( testInitializeParNums );
  CPPUNIT_TEST( testBackgroundFunction );
  CPPUNIT_TEST( testSetParameters );
  CPPUNIT_TEST_SUITE_END();
};

// Register the test suite in the registry.
// This way we will have to only pass the registry to the runner
// and it will contain all the registered test suites.
CPPUNIT_TEST_SUITE_REGISTRATION( TestBackgroundHandler );

#endif
