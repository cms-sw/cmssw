#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Luminosity/interface/LumiDetails.h"

#include <string>
#include <vector>
#include <iostream>

class TestLumiDetails: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestLumiDetails);  
  CPPUNIT_TEST(testConstructor);
  CPPUNIT_TEST(testFill);
  CPPUNIT_TEST_SUITE_END();
  
public:
  void setUp() {}
  void tearDown() {}

  void testConstructor();
  void testFill();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestLumiDetails);

void
TestLumiDetails::testConstructor() {
  std::cout << "\nTesting LumiDetails\n";
  LumiDetails lumiDetails;
  
  CPPUNIT_ASSERT(!lumiDetails.isValid());//so an empty detail array is not valid
  lumiDetails.setLumiVersion(std::string("v1"));
  CPPUNIT_ASSERT(lumiDetails.lumiVersion() == std::string("v1"));

  LumiDetails lumiDetails1(std::string("v2"));
  CPPUNIT_ASSERT(lumiDetails1.lumiVersion() == std::string("v2"));

  std::vector<std::string> const& v1 = lumiDetails.algoNames();
  std::vector<std::string> const& v2 = lumiDetails.algoNames();
  CPPUNIT_ASSERT(v2[0] == std::string("OCC1"));
  CPPUNIT_ASSERT(v2[1] == std::string("OCC2"));
  CPPUNIT_ASSERT(v2[2] == std::string("ET"));
  CPPUNIT_ASSERT(v2[3] == std::string("PLT"));
  CPPUNIT_ASSERT(v1.size() == 4U);
  CPPUNIT_ASSERT(v2.size() == 4U);
}

void
TestLumiDetails::testFill() {
  LumiDetails lumiDetails;
  std::vector<float> val;
  val.push_back(1.0f);
  val.push_back(2.0f);
  val.push_back(3.0f);

  std::vector<float> err;
  err.push_back(4.0f);
  err.push_back(5.0f);
  err.push_back(6.0f);
 
  std::vector<short> qual;
  qual.push_back(7);
  qual.push_back(8);
  qual.push_back(9);

  std::vector<float> beam1;
  beam1.push_back(10.0f);
  beam1.push_back(11.0f);
  beam1.push_back(12.0f);

  std::vector<float> beam2;
  beam2.push_back(13.0f);
  beam2.push_back(14.0f);
  beam2.push_back(15.0f);

  lumiDetails.fill(2, val, err, qual);
  lumiDetails.fillBeamIntensities(beam1, beam2);


  std::vector<float> val0;
  val0.push_back(1.0f);

  std::vector<float> err0;
  err0.push_back(4.0f);
 
  std::vector<short> qual0;
  qual0.push_back(7);

  lumiDetails.fill(0, val0, err0, qual0);

  std::vector<float> val1;
  std::vector<float> err1; 
  std::vector<short> qual1;
  lumiDetails.fill(1, val1, err1, qual1);

  std::vector<float> val3;
  val3.push_back(11.0f);
  val3.push_back(11.0f);

  std::vector<float> err3;
  err3.push_back(21.0f);
  err3.push_back(21.0f);
 
  std::vector<short> qual3;
  qual3.push_back(31);
  qual3.push_back(31);

  lumiDetails.fill(3, val3, err3, qual3);

  LumiDetails::ValueRange rangeVal = lumiDetails.lumiValuesForAlgo(2);
  std::cout << "values\n";
  int i = 1;
  for (std::vector<float>::const_iterator val = rangeVal.first;
       val != rangeVal.second; ++val, ++i) {
    std::cout << *val << " ";
    CPPUNIT_ASSERT(*val == i);
  }
  std::cout << "\n";
  CPPUNIT_ASSERT(lumiDetails.lumiValue(2,0) == 1.0f);
  CPPUNIT_ASSERT(lumiDetails.lumiValue(2,1) == 2.0f);
  CPPUNIT_ASSERT(lumiDetails.lumiValue(2,2) == 3.0f);

  LumiDetails::ErrorRange rangeErr = lumiDetails.lumiErrorsForAlgo(2);
  std::cout << "errors\n";
  i = 4;
  for (std::vector<float>::const_iterator err = rangeErr.first;
       err != rangeErr.second; ++err, ++i) {
    std::cout << *err << " ";
    CPPUNIT_ASSERT(*err == i);
  }
  std::cout << "\n";
  CPPUNIT_ASSERT(lumiDetails.lumiError(2,0) == 4.0f);
  CPPUNIT_ASSERT(lumiDetails.lumiError(2,1) == 5.0f);
  CPPUNIT_ASSERT(lumiDetails.lumiError(2,2) == 6.0f);

  LumiDetails::QualityRange rangeQual = lumiDetails.lumiQualitiesForAlgo(2);
  std::cout << "qualities\n";
  i = 7;
  for (std::vector<short>::const_iterator qual = rangeQual.first;
       qual != rangeQual.second; ++qual, ++i) {
    std::cout << *qual << " ";
    CPPUNIT_ASSERT(*qual == i);
  }
  std::cout << "\n";
  CPPUNIT_ASSERT(lumiDetails.lumiQuality(2,0) == 7);
  CPPUNIT_ASSERT(lumiDetails.lumiQuality(2,1) == 8);
  CPPUNIT_ASSERT(lumiDetails.lumiQuality(2,2) == 9);

  std::vector<float> const& beam1Intensities = lumiDetails.lumiBeam1Intensities();
  std::cout << "beam1Intensities\n";
  i = 10;
  for (std::vector<float>::const_iterator beam1 = beam1Intensities.begin(),
	                               beam1End = beam1Intensities.end();
       beam1 != beam1End; ++beam1, ++i) {
    std::cout << *beam1 << "\n";
    CPPUNIT_ASSERT(*beam1 == i);
  }
  std::cout << "\n";
  CPPUNIT_ASSERT(lumiDetails.lumiBeam1Intensity(0) == 10.0f);
  CPPUNIT_ASSERT(lumiDetails.lumiBeam1Intensity(1) == 11.0f);
  CPPUNIT_ASSERT(lumiDetails.lumiBeam1Intensity(2) == 12.0f);

  std::vector<float> const& beam2Intensities = lumiDetails.lumiBeam2Intensities();
  std::cout << "beam2Intensities\n";
  i = 13;
  for (std::vector<float>::const_iterator beam2 = beam2Intensities.begin(),
	                               beam2End = beam2Intensities.end();
       beam2 != beam2End; ++beam2, ++i) {
    std::cout << *beam2 << "\n";
    CPPUNIT_ASSERT(*beam2 == i);
  }
  std::cout << "\n";
  CPPUNIT_ASSERT(lumiDetails.lumiBeam2Intensity(0) == 13.0f);
  CPPUNIT_ASSERT(lumiDetails.lumiBeam2Intensity(1) == 14.0f);
  CPPUNIT_ASSERT(lumiDetails.lumiBeam2Intensity(2) == 15.0f);

  CPPUNIT_ASSERT(lumiDetails.isProductEqual(lumiDetails));

  LumiDetails lumiDetails2;
  CPPUNIT_ASSERT(!lumiDetails.isProductEqual(lumiDetails2));

  std::cout << lumiDetails;
}
