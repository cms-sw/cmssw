/*
 * Tests for running inference using custom thread pools.
 * Based on TensorFlow 2.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "testBase.h"

class testGraphLoading : public testBase {
  CPPUNIT_TEST_SUITE(testGraphLoading);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void checkAll() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testGraphLoading);

std::string testGraphLoading::pyScript() const { return "createconstantgraph.py"; }

void testGraphLoading::checkAll() {
  std::string pbFile = dataPath_ + "/constantgraph.pb";

  // initialize the TBB threadpool
  int nThreads = 4;
  tensorflow::TBBThreadPool::instance(nThreads);

  // load the graph
  tensorflow::setLogging();
  tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(pbFile);
  CPPUNIT_ASSERT(graphDef != nullptr);

  // create a new session and add the graphDef
  tensorflow::Session* session = tensorflow::createSession(graphDef);
  CPPUNIT_ASSERT(session != nullptr);

  // prepare inputs
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, 10});
  float* d = input.flat<float>().data();
  for (size_t i = 0; i < 10; i++, d++) {
    *d = float(i);
  }
  tensorflow::Tensor scale(tensorflow::DT_FLOAT, {});
  scale.scalar<float>()() = 1.0;

  // "no_threads" pool
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::run(session, {{"input", input}, {"scale", scale}}, {"output"}, &outputs, "no_threads");
  CPPUNIT_ASSERT(outputs.size() == 1);
  std::cout << outputs[0].DebugString() << std::endl;
  CPPUNIT_ASSERT(outputs[0].matrix<float>()(0, 0) == 46.);

  // "tbb" pool
  outputs.clear();
  tensorflow::run(session, {{"input", input}, {"scale", scale}}, {"output"}, &outputs, "tbb");
  CPPUNIT_ASSERT(outputs.size() == 1);
  std::cout << outputs[0].DebugString() << std::endl;
  CPPUNIT_ASSERT(outputs[0].matrix<float>()(0, 0) == 46.);

  // tensorflow defaut pool using a new session
  tensorflow::Session* session2 = tensorflow::createSession(graphDef, nThreads);
  CPPUNIT_ASSERT(session != nullptr);
  outputs.clear();
  tensorflow::run(session2, {{"input", input}, {"scale", scale}}, {"output"}, &outputs, "tensorflow");
  CPPUNIT_ASSERT(outputs.size() == 1);
  std::cout << outputs[0].DebugString() << std::endl;
  CPPUNIT_ASSERT(outputs[0].matrix<float>()(0, 0) == 46.);

  // force an exception
  CPPUNIT_ASSERT_THROW(
      tensorflow::run(session, {{"input", input}, {"scale", scale}}, {"output"}, &outputs, "not_existing"),
      cms::Exception);

  // cleanup
  CPPUNIT_ASSERT(tensorflow::closeSession(session));
  CPPUNIT_ASSERT(tensorflow::closeSession(session2));
  delete graphDef;
}
