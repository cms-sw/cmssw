#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TextOutputter.h>
#include "PhysicsTools/XGBoost/interface/XGBooster.h"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <xgboost/c_api.h>
#include <cmath>  // For NAN

class XGBoosterTest : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(XGBoosterTest);
  CPPUNIT_TEST(testModelFileNotFound);
  CPPUNIT_TEST(testSetAndPredict);
  CPPUNIT_TEST(testInvalidFeatureValue);
  CPPUNIT_TEST(testUnsetFeature);
  CPPUNIT_TEST(testPredict);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {
    std::string model_name = "test_model.xgb";

    // Prepare a test model
    generateTestModel(model_name);

    // Load the model
    std::ifstream modelFile(model_name);
    if (!modelFile.good()) {
      throw std::runtime_error("Model file not found");
    }
    booster = new pat::XGBooster(model_name);

    // Add features to the booster
    featureNames = {"feature1", "feature2", "feature3", "feature4"};
    for (const auto& featureName : featureNames) {
      booster->addFeature(featureName);
    }
  }

  void tearDown() override { delete booster; }

  //
  // Tests
  //

  void testModelFileNotFound() { CPPUNIT_ASSERT_THROW(pat::XGBooster("nonexistent_model.xgb"), std::runtime_error); }

  void testSetAndPredict() {
    booster->set(featureNames[0], 0.5);
    booster->set(featureNames[1], 1.2);
    booster->set(featureNames[2], -0.3);
    booster->set(featureNames[3], 0.8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, booster->predict(), 1e-5);
  }

  void testPredict() {
    std::vector<float> values = {0.5, 1.2, -0.3, 0.8};
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, booster->predict(values), 1e-5);
  }

  void testInvalidFeatureValue() {
    booster->set(featureNames[0], NAN);
    booster->set(featureNames[1], 1.2);
    booster->set(featureNames[2], -0.3);
    booster->set(featureNames[3], 0.8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-998, booster->predict(), 1e-5);
  }

  void testUnsetFeature() {
    booster->set(featureNames[0], 0.5);
    booster->set(featureNames[1], 1.2);
    booster->set(featureNames[2], -0.3);
    CPPUNIT_ASSERT_THROW(booster->predict(), std::runtime_error);
  }

private:
  pat::XGBooster* booster;
  std::vector<std::string> featureNames;

  void generateTestModel(const std::string& modelPath) {
    // Define a fixed dataset
    const int num_rows = 6;
    const int num_cols = 4;
    float data[num_rows * num_cols] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                                       13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float labels[num_rows] = {0, 1, 0, 1, 0, 1};

    // Create DMatrix
    DMatrixHandle dtrain;
    XGDMatrixCreateFromMat(data, num_rows, num_cols, NAN, &dtrain);
    XGDMatrixSetFloatInfo(dtrain, "label", labels, num_rows);

    // Set parameters
    BoosterHandle booster;
    XGBoosterCreate(&dtrain, 1, &booster);
    XGBoosterSetParam(booster, "max_depth", "2");
    XGBoosterSetParam(booster, "eta", "1");
    XGBoosterSetParam(booster, "objective", "binary:logistic");

    // Train the model
    for (int iter = 0; iter < 10; ++iter) {
      XGBoosterUpdateOneIter(booster, iter, dtrain);
    }

    // Save the model
    XGBoosterSaveModel(booster, modelPath.c_str());

    // Free memory
    XGBoosterFree(booster);
    XGDMatrixFree(dtrain);
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION(XGBoosterTest);

int main(int argc, char** argv) {
  CppUnit::TextUi::TestRunner runner;
  runner.addTest(XGBoosterTest::suite());

  // Add a progress listener
  CppUnit::BriefTestProgressListener progress;
  runner.eventManager().addListener(&progress);

  // Return 0 if all tests passed, 1 if there were any failures
  return runner.run() ? 0 : 1;
}
