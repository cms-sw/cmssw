/*
 * HelloWorld test of the TensorFlow interface.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <stdexcept>
#include <cppunit/extensions/HelperMacros.h>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "testBase.h"

class testHelloWorld : public testBase {
  CPPUNIT_TEST_SUITE(testHelloWorld);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testHelloWorld);

std::string testHelloWorld::pyScript() const { return "creategraph.py"; }

void testHelloWorld::test() {
  std::string modelDir = dataPath_ + "/simplegraph";
  // Testing CPU
  std::cout << "Testing CPU backend" << std::endl;
  tensorflow::Backend backend = tensorflow::Backend::cpu;

  // object to load and run the graph / session
  tensorflow::Status status;
  tensorflow::Options options{backend};
  tensorflow::setLogging();
  tensorflow::RunOptions runOptions;
  tensorflow::SavedModelBundle bundle;

  // load everything
  status = tensorflow::LoadSavedModel(options.getSessionOptions(), runOptions, modelDir, {"serve"}, &bundle);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    return;
  }

  // fetch the session
  tensorflow::Session* session = bundle.session.release();

  // prepare inputs
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, 10});
  float* d = input.flat<float>().data();
  for (size_t i = 0; i < 10; i++, d++) {
    *d = float(i);
  }
  tensorflow::Tensor scale(tensorflow::DT_FLOAT, {});
  scale.scalar<float>()() = 1.0;

  // prepare outputs
  std::vector<tensorflow::Tensor> outputs;

  // session run
  status = session->Run({{"input", input}, {"scale", scale}}, {"output"}, {}, &outputs);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    return;
  }

  // log the output tensor
  std::cout << outputs[0].DebugString() << std::endl;

  // close the session
  status = session->Close();
  if (!status.ok()) {
    std::cerr << "error while closing session" << std::endl;
  }
  delete session;
}
