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

#include "testBaseCUDA.h"

class testHelloWorldCUDA : public testBaseCUDA {
  CPPUNIT_TEST_SUITE(testHelloWorldCUDA);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testHelloWorldCUDA);

std::string testHelloWorldCUDA::pyScript() const { return "creategraph.py"; }

void testHelloWorldCUDA::test() {
  if (!cms::cudatest::testDevices())
    return;

  std::vector<edm::ParameterSet> psets;
  edm::ServiceToken serviceToken = edm::ServiceRegistry::createSet(psets);
  edm::ServiceRegistry::Operate operate(serviceToken);

  // Setup the CUDA Service
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::string const config = R"_(import FWCore.ParameterSet.Config as cms
process = cms.Process('Test')
process.add_(cms.Service('ResourceInformationService'))
process.add_(cms.Service('CUDAService'))
)_";
  std::unique_ptr<edm::ParameterSet> params;
  edm::makeParameterSets(config, params);
  edm::ServiceToken tempToken(edm::ServiceRegistry::createServicesFromConfig(std::move(params)));
  edm::ServiceRegistry::Operate operate2(tempToken);

  auto cs = makeCUDAService(edm::ParameterSet{});
  std::cout << "CUDA service enabled: " << cs.enabled() << std::endl;

  std::string modelDir = dataPath_ + "/simplegraph";
  // Testing CPU
  std::cout << "Testing CUDA backend" << std::endl;
  tensorflow::Backend backend = tensorflow::Backend::cuda;

  // object to load and run the graph / session
  tensorflow::Status status;
  tensorflow::SessionOptions sessionOptions;
  tensorflow::setBackend(sessionOptions, backend);
  tensorflow::RunOptions runOptions;
  tensorflow::SavedModelBundle bundle;

  // load everything
  status = tensorflow::LoadSavedModel(sessionOptions, runOptions, modelDir, {"serve"}, &bundle);
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
