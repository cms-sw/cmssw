/*
 * Tests for working with empty inputs
 *
 */

#include <stdexcept>
#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "testBase.h"

class testEmptyInputs : public testBase {
  CPPUNIT_TEST_SUITE(testEmptyInputs);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testEmptyInputs);

std::string testEmptyInputs::pyScript() const { return "createconstantgraph.py"; }

void testEmptyInputs::test() {
  std::string pbFile = dataPath_ + "/constantgraph.pb";

  std::cout << "Testing CPU backend" << std::endl;
  tensorflow::Backend backend = tensorflow::Backend::cpu;

  // load the graph
  tensorflow::Options options{backend};
  tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(pbFile);
  CPPUNIT_ASSERT(graphDef != nullptr);

  // create a new session and add the graphDef
  const tensorflow::Session* session = tensorflow::createSession(graphDef, options);
  CPPUNIT_ASSERT(session != nullptr);

  // example evaluation with empty tensor
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, 0});
  tensorflow::Tensor scale(tensorflow::DT_FLOAT, {});
  scale.scalar<float>()() = 1.0;
  std::vector<tensorflow::Tensor> outputs;

  // run using the convenience helper
  outputs.clear();
  tensorflow::run(session, {{"input", input}, {"scale", scale}}, {"output"}, &outputs);
  CPPUNIT_ASSERT(outputs.size() == 0);

  // cleanup
  CPPUNIT_ASSERT(tensorflow::closeSession(session));
  CPPUNIT_ASSERT(session == nullptr);
  delete graphDef;
}
