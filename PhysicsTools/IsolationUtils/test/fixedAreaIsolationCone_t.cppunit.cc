/*
 *  fixedAreaIsolationCone_t.cc
 *  
 *  Unit-test for FixedAreaIsolationCone::operator() 
 *
 *  Created by Chistian Veelken (UC Davis) on 15-Jan-09.
 *
 */

#include <memory>
#include <iomanip>

#include <cppunit/extensions/HelperMacros.h>

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>

#include <TMath.h>

#include "PhysicsTools/IsolationUtils/interface/FixedAreaIsolationCone.h"

#include "PhysicsTools/IsolationUtils/interface/IntegralOverPhiFunction.h"
#include "PhysicsTools/IsolationUtils/interface/IntegrandThetaFunction.h"
#include "Math/BrentMethods.h"


class testFixedAreaIsolationCone : public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testFixedAreaIsolationCone);

CPPUNIT_TEST(operatorTest);

CPPUNIT_TEST_SUITE_END();
public:
  void operatorTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testFixedAreaIsolationCone);

const unsigned numEta = 7;
const double eta[numEta] = { -2.5, -1.5, -0.5, 0., +0.5, +1.5, + 2.5 };
const double signalConeSize = 0.07;
const double isolationConeArea = 0.7;
const double isolationConeSizeRef[numEta] = { 0.1545, 0.2014, 0.4129, 0.4682, 0.4129, 0.2014, 0.1545 };
const unsigned numPhi = 16;
const double requiredRelativePrecision = 1.e-3;

void testFixedAreaIsolationCone::operatorTest()
{
  FixedAreaIsolationCone fixedAreaIsolationCone;
  fixedAreaIsolationCone.setAcceptanceLimit(2.5);

  for ( unsigned iEta = 0; iEta < numEta; ++iEta ) {
    double theta = 2*TMath::ATan(TMath::Exp(-eta[iEta]));
    //std::cout << "theta = " << theta*180./TMath::Pi() << std::endl;

    for ( unsigned iPhi = 0; iPhi < numPhi; ++iPhi ) {
      double phi = TMath::Pi() - iPhi*2*TMath::Pi()/numPhi;
      //std::cout << " phi = " << phi*180./TMath::Pi() << ":";

      int error = 0;
      double isolationConeSize = fixedAreaIsolationCone(theta, phi, signalConeSize, isolationConeArea, error);
      //std::cout << " isolationConeSize = " << isolationConeSize << std::endl;
      //std::cout << " error = " << error << std::endl;

      CPPUNIT_ASSERT(error == 0);
      CPPUNIT_ASSERT(TMath::Abs(isolationConeSize - isolationConeSizeRef[iEta]) < (requiredRelativePrecision * isolationConeSizeRef[iEta]));
    }
  }
}
