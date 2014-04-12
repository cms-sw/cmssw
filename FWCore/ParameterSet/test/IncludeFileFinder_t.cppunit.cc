/*
 */

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/ParameterSet/interface/IncludeFileFinder.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

using std::string;
using std::vector;
using edm::pset::IncludeFileFinder;
using std::pair;

class IncludeFileFinderTest: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(IncludeFileFinderTest);
  CPPUNIT_TEST(strippingTest);
  CPPUNIT_TEST(twoWordsTest);
  CPPUNIT_TEST(ultimateTest);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){
    if(not edmplugin::PluginManager::isAvailable()) {
      edmplugin::PluginManager::configure(edmplugin::standard::config());
    }
  }
  void tearDown(){}

  void strippingTest();
  void stripTrailerTest();
  void twoWordsTest();
  void ultimateTest();
private:
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(IncludeFileFinderTest);


void IncludeFileFinderTest::strippingTest()
{
  string original = "libSimCalorimetryCaloSimAlgosPlugins.so";

  string result = IncludeFileFinder::stripHeader(original);
  CPPUNIT_ASSERT (result == "SimCalorimetryCaloSimAlgosPlugins.so");
   
  string result2 =  IncludeFileFinder::stripTrailer(result);
  CPPUNIT_ASSERT (result2 == "SimCalorimetryCaloSimAlgos");
}


void IncludeFileFinderTest::twoWordsTest()
{
  string original = "SimCalorimetryCaloSimAlgos";
  vector<pair<string, string> > twoWords = IncludeFileFinder::twoWordsFrom(original);

  CPPUNIT_ASSERT (twoWords.size() == 4);
  CPPUNIT_ASSERT (twoWords[0].first  == "Sim");
  CPPUNIT_ASSERT (twoWords[0].second == "CalorimetryCaloSimAlgos");
  CPPUNIT_ASSERT (twoWords[1].first  == "SimCalorimetry");
  CPPUNIT_ASSERT (twoWords[1].second == "CaloSimAlgos");
  CPPUNIT_ASSERT (twoWords[2].first  == "SimCalorimetryCalo");
  CPPUNIT_ASSERT (twoWords[2].second == "SimAlgos");
  CPPUNIT_ASSERT (twoWords[3].first  == "SimCalorimetryCaloSim");
  CPPUNIT_ASSERT (twoWords[3].second == "Algos");

}


void IncludeFileFinderTest::ultimateTest()
{
  string moduleClass = "CSCDigiProducer";
  IncludeFileFinder finder;
//  string library = finder.libraryOf(moduleClass);
//  CPPUNIT_ASSERT (library == "libSimMuonCSCDigitizer.so");

  string moduleLabel = "muoncscdigi";
//  edm::FileInPath file = finder.find(moduleClass, moduleLabel);
  // will throw if can't find
}

