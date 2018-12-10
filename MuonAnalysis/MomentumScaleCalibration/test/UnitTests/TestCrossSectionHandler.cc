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

#include "MuonAnalysis/MomentumScaleCalibration/interface/CrossSectionHandler.h"

#ifndef TestCrossSectionHandler_cc
#define TestCrossSectionHandler_cc

class TestCrossSectionHandler : public CppUnit::TestFixture {
public:
  TestCrossSectionHandler() {}
  void setUp()
  {
    crossSection.push_back(1.233);
    crossSection.push_back(2.07);
    crossSection.push_back(6.33);
    crossSection.push_back(13.9);
    crossSection.push_back(2.169);
    crossSection.push_back(127.2);

    // First case: fit all the Upsilons
    fill_n(back_inserter(resfind_1), 6, 0);
    resfind_1[1] = 1;
    resfind_1[2] = 1;
    resfind_1[3] = 1;
    crossSectionHandler_1 = new CrossSectionHandler(crossSection, resfind_1);

    // Second case: fit only the Upsilon(3S)
    fill_n(back_inserter(resfind_2), 6, 0);
    resfind_2[1] = 1;
    crossSectionHandler_2 = new CrossSectionHandler(crossSection, resfind_2);

    // Third case: fit the Upsilon(3S) and the J/Psi and Psi(2S)
    fill_n(back_inserter(resfind_3), 6, 0);
    resfind_3[1] = 1;
    resfind_3[4] = 1;
    resfind_3[5] = 1;
    crossSectionHandler_3 = new CrossSectionHandler(crossSection, resfind_3);

    // Fourth case: fit the Upsilon(3S) and (2S) and the J/Psi and Psi(2S)
    fill_n(back_inserter(resfind_4), 6, 0);
    resfind_4[1] = 1;
    resfind_4[2] = 1;
    resfind_4[4] = 1;
    resfind_4[5] = 1;
    crossSectionHandler_4 = new CrossSectionHandler(crossSection, resfind_4);

    // Fifth case: fit nothing
    fill_n(back_inserter(resfind_5), 6, 0);
    crossSectionHandler_5 = new CrossSectionHandler(crossSection, resfind_5);

    // Sixth case: fit everything
    fill_n(back_inserter(resfind_6), 6, 1);
    crossSectionHandler_6 = new CrossSectionHandler(crossSection, resfind_6);
  }

  void tearDown()
  {
    delete crossSectionHandler_1;
    delete crossSectionHandler_2;
    delete crossSectionHandler_3;
    delete crossSectionHandler_4;
    delete crossSectionHandler_5;
    delete crossSectionHandler_6;
  }

  void testConstructor()
  {
    CPPUNIT_ASSERT( crossSectionHandler_1->parNum_ == 2 );
    CPPUNIT_ASSERT( crossSectionHandler_2->parNum_ == 0 );
    CPPUNIT_ASSERT( crossSectionHandler_3->parNum_ == 2 );
    CPPUNIT_ASSERT( crossSectionHandler_4->parNum_ == 3 );
    CPPUNIT_ASSERT( crossSectionHandler_5->parNum_ == 0 );
    CPPUNIT_ASSERT( crossSectionHandler_6->parNum_ == 5 );
    CPPUNIT_ASSERT( crossSectionHandler_1->vars_.size() == crossSectionHandler_1->parNum_ );
    CPPUNIT_ASSERT( crossSectionHandler_2->vars_.size() == crossSectionHandler_2->parNum_ );
    CPPUNIT_ASSERT( crossSectionHandler_3->vars_.size() == crossSectionHandler_3->parNum_ );
    CPPUNIT_ASSERT( crossSectionHandler_4->vars_.size() == crossSectionHandler_4->parNum_ );
    CPPUNIT_ASSERT( crossSectionHandler_5->vars_.size() == crossSectionHandler_5->parNum_ );
    CPPUNIT_ASSERT( crossSectionHandler_6->vars_.size() == crossSectionHandler_6->parNum_ );
  }

  void testComputeRelativeCrossSections()
  {
    crossSectionHandler_1->computeRelativeCrossSections(crossSection, resfind_1);
    crossSectionHandler_2->computeRelativeCrossSections(crossSection, resfind_2);
    crossSectionHandler_3->computeRelativeCrossSections(crossSection, resfind_3);
    crossSectionHandler_4->computeRelativeCrossSections(crossSection, resfind_4);
    crossSectionHandler_5->computeRelativeCrossSections(crossSection, resfind_5);
    crossSectionHandler_6->computeRelativeCrossSections(crossSection, resfind_6);

    std::vector<std::vector<double> > relativeCrossSections;
    relativeCrossSections.push_back(expandCrossSectionVec(crossSectionHandler_1->relativeCrossSectionVec_, resfind_1));
    relativeCrossSections.push_back(expandCrossSectionVec(crossSectionHandler_2->relativeCrossSectionVec_, resfind_2));
    relativeCrossSections.push_back(expandCrossSectionVec(crossSectionHandler_3->relativeCrossSectionVec_, resfind_3));
    relativeCrossSections.push_back(expandCrossSectionVec(crossSectionHandler_4->relativeCrossSectionVec_, resfind_4));
    relativeCrossSections.push_back(expandCrossSectionVec(crossSectionHandler_5->relativeCrossSectionVec_, resfind_5));
    relativeCrossSections.push_back(expandCrossSectionVec(crossSectionHandler_6->relativeCrossSectionVec_, resfind_6));

    checkRelativeCrossSections(relativeCrossSections);
  }

  void checkRelativeCrossSections(const std::vector<std::vector<double> > & relativeCrossSections)
  {
    // First case: fit all the Upsilons
    double norm = crossSection[1] + crossSection[2] + crossSection[3];
    CPPUNIT_ASSERT( relativeCrossSections[0].size() == 6 );
    CPPUNIT_ASSERT( float(relativeCrossSections[0][0]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[0][1]) == float(crossSection[1]/norm) );
    CPPUNIT_ASSERT( float(relativeCrossSections[0][2]) == float(crossSection[2]/norm) );
    CPPUNIT_ASSERT( float(relativeCrossSections[0][3]) == float(crossSection[3]/norm) );
    CPPUNIT_ASSERT( float(relativeCrossSections[0][4]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[0][5]) == float(0.) );

    // Second case: fit only the Upsilon(3S)
    CPPUNIT_ASSERT( relativeCrossSections[1].size() == 6 );
    CPPUNIT_ASSERT( float(relativeCrossSections[1][0]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[1][1]) == float(1.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[1][2]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[1][3]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[1][4]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[1][5]) == float(0.) );

    // Third case: fit the Upsilon(3S) and the J/Psi and Psi(2S)
    CPPUNIT_ASSERT( relativeCrossSections[2].size() == 6 );
    norm = crossSection[1] + crossSection[4] + crossSection[5];
    CPPUNIT_ASSERT( float(relativeCrossSections[2][0]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[2][1]) == float(crossSection[1]/norm) );
    CPPUNIT_ASSERT( float(relativeCrossSections[2][2]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[2][3]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[2][4]) == float(crossSection[4]/norm) );
    CPPUNIT_ASSERT( float(relativeCrossSections[2][5]) == float(crossSection[5]/norm) );

    // Fourth case: fit the Upsilon(3S) and (2S) and the J/Psi and Psi(2S)
    CPPUNIT_ASSERT( relativeCrossSections[3].size() == 6 );
    norm = crossSection[1] + crossSection[2] + crossSection[4] + crossSection[5];
    CPPUNIT_ASSERT( float(relativeCrossSections[3][0]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[3][1]) == float(crossSection[1]/norm) );
    CPPUNIT_ASSERT( float(relativeCrossSections[3][2]) == float(crossSection[2]/norm) );
    CPPUNIT_ASSERT( float(relativeCrossSections[3][3]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[3][4]) == float(crossSection[4]/norm) );
    CPPUNIT_ASSERT( float(relativeCrossSections[3][5]) == float(crossSection[5]/norm) );

    // Fifth case: fit nothing
    CPPUNIT_ASSERT( relativeCrossSections[4].size() == 6 );
    CPPUNIT_ASSERT( float(relativeCrossSections[4][0]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[4][1]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[4][2]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[4][3]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[4][4]) == float(0.) );
    CPPUNIT_ASSERT( float(relativeCrossSections[4][5]) == float(0.) );

    // Sixth case: fit everything
    CPPUNIT_ASSERT( relativeCrossSections[5].size() == 6 );
    norm = crossSection[0] + crossSection[1] + crossSection[2] + crossSection[3] + crossSection[4] + crossSection[5];
    CPPUNIT_ASSERT( float(relativeCrossSections[5][0]) == float(crossSection[0]/norm) );
    CPPUNIT_ASSERT( float(relativeCrossSections[5][1]) == float(crossSection[1]/norm) );
    CPPUNIT_ASSERT( float(relativeCrossSections[5][2]) == float(crossSection[2]/norm) );
    CPPUNIT_ASSERT( float(relativeCrossSections[5][3]) == float(crossSection[3]/norm) );
    CPPUNIT_ASSERT( float(relativeCrossSections[5][4]) == float(crossSection[4]/norm) );
    CPPUNIT_ASSERT( float(relativeCrossSections[5][5]) == float(crossSection[5]/norm) );
  }

  void testImposeConstraint()
  {
    // First case: fit all the Upsilons
    crossSectionHandler_1->computeRelativeCrossSections(crossSection, resfind_1);
    crossSectionHandler_1->imposeConstraint();
    CPPUNIT_ASSERT( float(crossSectionHandler_1->vars_[0]) == float(crossSection[2]/crossSection[1]) );
    CPPUNIT_ASSERT( float(crossSectionHandler_1->vars_[1]) == float(crossSection[3]/crossSection[2]) );

    // Second case: fit only the Upsilon(3S). Nothing to test. The vars_ vector is empty (tested in the constructor test)

    // Third case: fit the Upsilon(3S) and the J/Psi and Psi(2S)
    crossSectionHandler_3->computeRelativeCrossSections(crossSection, resfind_3);
    crossSectionHandler_3->imposeConstraint();
    CPPUNIT_ASSERT( float(crossSectionHandler_3->vars_[0]) == float(crossSection[4]/crossSection[1]) );
    CPPUNIT_ASSERT( float(crossSectionHandler_3->vars_[1]) == float(crossSection[5]/crossSection[4]) );

    // Fourth case: fit the Upsilon(3S) and (2S) and the J/Psi and Psi(2S)
    crossSectionHandler_4->computeRelativeCrossSections(crossSection, resfind_4);
    crossSectionHandler_4->imposeConstraint();
    CPPUNIT_ASSERT( float(crossSectionHandler_4->vars_[0]) == float(crossSection[2]/crossSection[1]) );
    CPPUNIT_ASSERT( float(crossSectionHandler_4->vars_[1]) == float(crossSection[4]/crossSection[2]) );
    CPPUNIT_ASSERT( float(crossSectionHandler_4->vars_[2]) == float(crossSection[5]/crossSection[4]) );

    // Fifth case: fit nothing. vars_.size() = 0 tested in the constructor test.

    // Sixth case: fit everything
    crossSectionHandler_6->computeRelativeCrossSections(crossSection, resfind_6);
    crossSectionHandler_6->imposeConstraint();
    CPPUNIT_ASSERT( float(crossSectionHandler_6->vars_[0]) == float(crossSection[1]/crossSection[0]) );
    CPPUNIT_ASSERT( float(crossSectionHandler_6->vars_[1]) == float(crossSection[2]/crossSection[1]) );
    CPPUNIT_ASSERT( float(crossSectionHandler_6->vars_[2]) == float(crossSection[3]/crossSection[2]) );
    CPPUNIT_ASSERT( float(crossSectionHandler_6->vars_[3]) == float(crossSection[4]/crossSection[3]) );
    CPPUNIT_ASSERT( float(crossSectionHandler_6->vars_[4]) == float(crossSection[5]/crossSection[4]) );
  }

  void testRelativeCrossSections()
  {
    std::vector<std::vector<double> > relativeCrossSections;
    relativeCrossSections.push_back( getRelativeCrossSections(crossSectionHandler_1, resfind_1) );
    relativeCrossSections.push_back( getRelativeCrossSections(crossSectionHandler_2, resfind_2) );
    relativeCrossSections.push_back( getRelativeCrossSections(crossSectionHandler_3, resfind_3) );
    relativeCrossSections.push_back( getRelativeCrossSections(crossSectionHandler_4, resfind_4) );
    relativeCrossSections.push_back( getRelativeCrossSections(crossSectionHandler_5, resfind_5) );
    relativeCrossSections.push_back( getRelativeCrossSections(crossSectionHandler_6, resfind_6) );

    checkRelativeCrossSections( relativeCrossSections );
  }

  std::vector<double> getRelativeCrossSections(CrossSectionHandler * crossSectionHandler, const std::vector<int> resfind)
  {
    crossSectionHandler->computeRelativeCrossSections(crossSection, resfind);
    crossSectionHandler->imposeConstraint();
    std::vector<double>::const_iterator it = crossSectionHandler->vars_.begin();
    double * variables = new double[crossSectionHandler->vars_.size()];
    unsigned int i = 0;
    for( ; it != crossSectionHandler->vars_.end(); ++it, ++i ) {
      variables[i] = *it;
    }
    std::vector<double> vars(crossSectionHandler->relativeCrossSections(variables, resfind));

    delete[] variables;
    return vars;
  }

  void testSetParameters()
  {
    TMinuit rmin(crossSectionHandler_1->parNum_);
    std::vector<int> crossSectionOrder(6, 0);
    crossSectionOrder[1] = 1;

    crossSectionHandler_1->setParameters( Start, Step, Mini, Maxi, ind, parname, crossSection, crossSectionOrder, resfind_1 );
    CPPUNIT_ASSERT(Start[0] == crossSectionHandler_1->vars_[0]);
    CPPUNIT_ASSERT(Start[1] == crossSectionHandler_1->vars_[1]);
    CPPUNIT_ASSERT(parname[0] == "cross section var 1");
    CPPUNIT_ASSERT(parname[1] == "cross section var 2");
    CPPUNIT_ASSERT(ind[0] == 0);
    CPPUNIT_ASSERT(ind[1] == 1);
  }

  void testReleaseParameters()
  {
    TMinuit rmin(crossSectionHandler_1->parNum_);
    std::vector<int> crossSectionOrder(6, 0);
    crossSectionOrder[1] = 1;
    crossSectionHandler_1->setParameters( Start, Step, Mini, Maxi, ind, parname, crossSection, crossSectionOrder, resfind_1 );
    int ierror;
    for( unsigned int ipar=0; ipar<crossSectionHandler_1->parNum(); ++ipar ) {
      rmin.mnparm( ipar, parname[ipar], Start[ipar], Step[ipar], Mini[ipar], Maxi[ipar], ierror );
      rmin.FixParameter(ipar);
    }

    std::vector<int> parfix(6, 0);
    // Test by setting two parameters to order 1, but one is for the Z which is not fitted and will be ignored.
    int * ind = new int[6];
    ind[0] = 0;
    ind[1] = 1;
    ind[2] = 0;
    ind[3] = 0;
    ind[4] = 0;
    ind[5] = 1;
    crossSectionHandler_1->releaseParameters( rmin, resfind_1, parfix, ind, 0, 0 );
    CPPUNIT_ASSERT(rmin.GetNumFixedPars() == 1);
    crossSectionHandler_1->releaseParameters( rmin, resfind_1, parfix, ind, 1, 0 );
    CPPUNIT_ASSERT(rmin.GetNumFixedPars() == 0);
  }

  std::vector<double> expandCrossSectionVec( const std::vector<double> & relativeCrossSectionVec, const std::vector<int> & resfind )
  {
    std::vector<double> relCrossSec;
    unsigned int smallerVectorIndex = 0;
    std::vector<int>::const_iterator it = resfind.begin();
    for( ; it != resfind.end(); ++it ) {
      if( *it == 0 ) {
        relCrossSec.push_back(0.);
      }
      else {
        relCrossSec.push_back(relativeCrossSectionVec[smallerVectorIndex]);
        ++smallerVectorIndex;
      }
    }
    return relCrossSec;
  }


  // Data members
  CrossSectionHandler * crossSectionHandler_1;
  CrossSectionHandler * crossSectionHandler_2;
  CrossSectionHandler * crossSectionHandler_3;
  CrossSectionHandler * crossSectionHandler_4;
  CrossSectionHandler * crossSectionHandler_5;
  CrossSectionHandler * crossSectionHandler_6;
  std::vector<int> resfind_1;
  std::vector<int> resfind_2;
  std::vector<int> resfind_3;
  std::vector<int> resfind_4;
  std::vector<int> resfind_5;
  std::vector<int> resfind_6;
  std::vector<double> crossSection;

  double Start[100];
  double Step[100];
  double Mini[100];
  double Maxi[100];
  int ind[100];
  TString parname[100];

  // Declare and build the test suite
  CPPUNIT_TEST_SUITE( TestCrossSectionHandler );
  CPPUNIT_TEST( testConstructor );
  CPPUNIT_TEST( testComputeRelativeCrossSections );
  CPPUNIT_TEST( testImposeConstraint );
  CPPUNIT_TEST( testRelativeCrossSections );
  CPPUNIT_TEST( testSetParameters );
  CPPUNIT_TEST( testReleaseParameters );
  CPPUNIT_TEST_SUITE_END();
};

// Register the test suite in the registry.
// This way we will have to only pass the registry to the runner
// and it will contain all the registered test suites.
CPPUNIT_TEST_SUITE_REGISTRATION( TestCrossSectionHandler );

#endif
