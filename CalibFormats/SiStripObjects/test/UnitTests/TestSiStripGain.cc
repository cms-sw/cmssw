#include <cppunit/CompilerOutputter.h>
#include <cppunit/TestFixture.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TestRunner.h>
#include <cppunit/TextTestProgressListener.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>

#include <algorithm>
#include <boost/foreach.hpp>
#include <iterator>

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#ifndef TestSiStripGain_cc
#define TestSiStripGain_cc

class TestSiStripGain : public CppUnit::TestFixture {
public:
  TestSiStripGain() {}
  void setUp() {
    detId = 436282904;

    apvGain1 = new SiStripApvGain;
    std::vector<float> theSiStripVector;
    theSiStripVector.push_back(1.);
    theSiStripVector.push_back(0.8);
    theSiStripVector.push_back(1.2);
    theSiStripVector.push_back(2.);
    fillApvGain(apvGain1, detId, theSiStripVector);

    apvGain2 = new SiStripApvGain;
    theSiStripVector.clear();
    theSiStripVector.push_back(1.);
    theSiStripVector.push_back(1. / 0.8);
    theSiStripVector.push_back(1. / 1.2);
    theSiStripVector.push_back(2.);
    fillApvGain(apvGain2, detId, theSiStripVector);
  }

  void tearDown() {
    delete apvGain1;
    delete apvGain2;
  }

  void testConstructor() {
    // Test with normalization factor = 1
    apvGainsTest(1.);
    // Test with normalization factor != 1
    apvGainsTest(2.);
  }

  void testMultiply() {
    // Test with norm = 1
    multiplyTest(1., 1.);

    // Test with norm != 1
    multiplyTest(2., 3.);
  }

  void multiplyTest(const float &norm1, const float &norm2) {
    std::pair<std::string, std::string> recordLabelPair1("gainRcd1", "");
    std::pair<std::string, std::string> recordLabelPair2("gainRcd2", "");
    std::vector<std::pair<std::string, std::string>> recordLabelPairVector;
    recordLabelPairVector.push_back(recordLabelPair1);
    recordLabelPairVector.push_back(recordLabelPair2);
    std::vector<float> normVector;
    normVector.push_back(norm1);
    normVector.push_back(norm2);

    const auto detInfo =
        SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());

    SiStripGain gain(*apvGain1, norm1, recordLabelPair1, detInfo);
    gain.multiply(*apvGain2, norm2, recordLabelPair2, detInfo);
    SiStripApvGain::Range range = gain.getRange(detId);

    // Check multiplication
    CPPUNIT_ASSERT(float(gain.getApvGain(0, range)) == float(1. / norm1 * 1. / norm2));
    CPPUNIT_ASSERT(float(gain.getApvGain(1, range)) == float(0.8 / norm1 * 1. / (0.8 * norm2)));
    CPPUNIT_ASSERT(float(gain.getApvGain(2, range)) == float(1.2 / norm1 * 1. / (1.2 * norm2)));
    CPPUNIT_ASSERT(float(gain.getApvGain(3, range)) == float(2. / norm1 * 2. / norm2));

    checkTag(gain, normVector, normVector.size(), recordLabelPairVector);
  }

  void apvGainsTest(const float &norm) {
    const auto detInfo =
        SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());

    SiStripGain gain(*apvGain1, norm, detInfo);
    SiStripApvGain::Range range = gain.getRange(detId);
    CPPUNIT_ASSERT(float(gain.getApvGain(0, range)) == float(1. / norm));
    CPPUNIT_ASSERT(float(gain.getApvGain(1, range)) == float(0.8 / norm));
    CPPUNIT_ASSERT(float(gain.getApvGain(2, range)) == float(1.2 / norm));
    CPPUNIT_ASSERT(float(gain.getApvGain(3, range)) == float(2. / norm));
    checkTag(gain, norm, "", "");

    SiStripGain gain2(*apvGain2, norm, detInfo);
    SiStripApvGain::Range range2 = gain2.getRange(detId);
    CPPUNIT_ASSERT(float(gain2.getApvGain(0, range2)) == float(1. / norm));
    CPPUNIT_ASSERT(float(gain2.getApvGain(1, range2)) == float(1. / (norm * 0.8)));
    CPPUNIT_ASSERT(float(gain2.getApvGain(2, range2)) == float(1. / (norm * 1.2)));
    CPPUNIT_ASSERT(float(gain2.getApvGain(3, range2)) == float(2. / norm));
    checkTag(gain2, norm, "", "");
  }

  void checkTag(const SiStripGain &gain, const float &norm, const std::string &rcdName, const std::string &labelName) {
    std::vector<float> normVector;
    normVector.push_back(norm);
    std::vector<std::pair<std::string, std::string>> recordLabelPairVector;
    recordLabelPairVector.push_back(std::make_pair(rcdName, labelName));
    checkTag(gain, normVector, 1, recordLabelPairVector);
  }
  void checkTag(const SiStripGain &gain,
                const std::vector<float> &norm,
                const uint32_t tagNum,
                const std::vector<std::pair<std::string, std::string>> &recordLabelPair) {
    CPPUNIT_ASSERT(gain.getNumberOfTags() == tagNum);
    for (unsigned int i = 0; i < tagNum; ++i) {
      CPPUNIT_ASSERT(float(gain.getTagNorm(i)) == norm[i]);
      CPPUNIT_ASSERT(gain.getRcdName(i) == recordLabelPair[i].first);
      CPPUNIT_ASSERT(gain.getLabelName(i) == recordLabelPair[i].second);
    }
  }

  void fillApvGain(SiStripApvGain *apvGain, const uint32_t detId, const std::vector<float> &theSiStripVector) {
    SiStripApvGain::Range range(theSiStripVector.begin(), theSiStripVector.end());
    apvGain->put(detId, range);
  }

  SiStripApvGain *apvGain1;
  SiStripApvGain *apvGain2;
  uint32_t detId;

  // Declare and build the test suite
  CPPUNIT_TEST_SUITE(TestSiStripGain);
  CPPUNIT_TEST(testConstructor);
  CPPUNIT_TEST(testMultiply);
  CPPUNIT_TEST_SUITE_END();
};

// Register the test suite in the registry.
// This way we will have to only pass the registry to the runner
// and it will contain all the registered test suites.
CPPUNIT_TEST_SUITE_REGISTRATION(TestSiStripGain);

#endif
