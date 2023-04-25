/*
 * Tests for loading meta graphs via the SavedModel interface.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <stdexcept>
#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "testBase.h"

class testMetaGraphLoading : public testBase {
  CPPUNIT_TEST_SUITE(testMetaGraphLoading);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void checkAll() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testMetaGraphLoading);

std::string testMetaGraphLoading::pyScript() const { return "creategraph.py"; }

void testMetaGraphLoading::checkAll() {
  std::string exportDir = dataPath_ + "/simplegraph";

  // load the graph
  tensorflow::setLogging();
  tensorflow::MetaGraphDef* metaGraphDef = tensorflow::loadMetaGraphDef(exportDir);
  CPPUNIT_ASSERT(metaGraphDef != nullptr);

  // create a new, empty session
  tensorflow::Session* session1 = tensorflow::createSession();
  CPPUNIT_ASSERT(session1 != nullptr);

  // create a new session, using the meta graph
  tensorflow::Session* session2 = tensorflow::createSession(metaGraphDef, exportDir);
  CPPUNIT_ASSERT(session2 != nullptr);

  // check for exception
  CPPUNIT_ASSERT_THROW(tensorflow::createSession(nullptr, exportDir), cms::Exception);

  // example evaluation
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, 10});
  float* d = input.flat<float>().data();
  for (size_t i = 0; i < 10; i++, d++) {
    *d = float(i);
  }
  tensorflow::Tensor scale(tensorflow::DT_FLOAT, {});
  scale.scalar<float>()() = 1.0;

  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status status = session2->Run({{"input", input}, {"scale", scale}}, {"output"}, {}, &outputs);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    CPPUNIT_ASSERT(false);
  }

  // check the output
  CPPUNIT_ASSERT(outputs.size() == 1);
  std::cout << outputs[0].DebugString() << std::endl;
  CPPUNIT_ASSERT(outputs[0].matrix<float>()(0, 0) == 46.);

  // run again using the convenience helper
  outputs.clear();
  tensorflow::run(session2, {{"input", input}, {"scale", scale}}, {"output"}, &outputs);
  CPPUNIT_ASSERT(outputs.size() == 1);
  std::cout << outputs[0].DebugString() << std::endl;
  CPPUNIT_ASSERT(outputs[0].matrix<float>()(0, 0) == 46.);

  // check for exception
  CPPUNIT_ASSERT_THROW(tensorflow::run(session2, {{"foo", input}}, {"output"}, &outputs), cms::Exception);

  // cleanup
  CPPUNIT_ASSERT(tensorflow::closeSession(session1));
  CPPUNIT_ASSERT(tensorflow::closeSession(session2));
  delete metaGraphDef;
}
