/*
 * Tests for working with constant sessions.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <stdexcept>
#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "testBaseCUDA.h"

class testConstSessionCUDA : public testBaseCUDA {
  CPPUNIT_TEST_SUITE(testConstSessionCUDA);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testConstSessionCUDA);

std::string testConstSessionCUDA::pyScript() const { return "createconstantgraph.py"; }

void testConstSessionCUDA::test() {
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
  edm::Service<CUDAInterface> cuda;
  std::cout << "CUDA service enabled: " << cuda->enabled() << std::endl;

  std::cout << "Testing CUDA backend" << std::endl;
  tensorflow::Backend backend = tensorflow::Backend::cuda;

  // load the graph
  std::string pbFile = dataPath_ + "/constantgraph.pb";
  tensorflow::setLogging();
  tensorflow::Options options{backend};

  tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(pbFile);
  CPPUNIT_ASSERT(graphDef != nullptr);

  // create a new session and add the graphDef
  const tensorflow::Session* session = tensorflow::createSession(graphDef, options);
  CPPUNIT_ASSERT(session != nullptr);

  // example evaluation
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, 10});
  float* d = input.flat<float>().data();
  for (size_t i = 0; i < 10; i++, d++) {
    *d = float(i);
  }
  tensorflow::Tensor scale(tensorflow::DT_FLOAT, {});
  scale.scalar<float>()() = 1.0;
  std::vector<tensorflow::Tensor> outputs;

  // run using the convenience helper
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
