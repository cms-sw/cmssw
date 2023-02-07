/*
 * Tests for loading graphs via the converted protobuf files.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <stdexcept>
#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "testBaseCUDA.h"

class testGraphLoadingCUDA : public testBaseCUDA {
  CPPUNIT_TEST_SUITE(testGraphLoadingCUDA);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testGraphLoadingCUDA);

std::string testGraphLoadingCUDA::pyScript() const { return "createconstantgraph.py"; }

void testGraphLoadingCUDA::test() {
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

  std::string pbFile = dataPath_ + "/constantgraph.pb";

  std::cout << "Testing CUDA backend" << std::endl;
  tensorflow::Backend backend = tensorflow::Backend::cuda;

  // load the graph
  tensorflow::setLogging();
  tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(pbFile);
  CPPUNIT_ASSERT(graphDef != nullptr);

  // create a new session and add the graphDef
  tensorflow::Session* session = tensorflow::createSession(graphDef, backend);
  CPPUNIT_ASSERT(session != nullptr);

  // check for exception
  CPPUNIT_ASSERT_THROW(tensorflow::createSession(nullptr, backend), cms::Exception);

  // example evaluation
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, 10});
  float* d = input.flat<float>().data();
  for (size_t i = 0; i < 10; i++, d++) {
    *d = float(i);
  }
  tensorflow::Tensor scale(tensorflow::DT_FLOAT, {});
  scale.scalar<float>()() = 1.0;

  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status status = session->Run({{"input", input}, {"scale", scale}}, {"output"}, {}, &outputs);
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
  tensorflow::run(session, {{"input", input}, {"scale", scale}}, {"output"}, &outputs);
  CPPUNIT_ASSERT(outputs.size() == 1);
  std::cout << outputs[0].DebugString() << std::endl;
  CPPUNIT_ASSERT(outputs[0].matrix<float>()(0, 0) == 46.);

  // check for exception
  CPPUNIT_ASSERT_THROW(tensorflow::run(session, {{"foo", input}}, {"output"}, &outputs), cms::Exception);

  // cleanup
  CPPUNIT_ASSERT(tensorflow::closeSession(session));
  CPPUNIT_ASSERT(session == nullptr);
  delete graphDef;
}
