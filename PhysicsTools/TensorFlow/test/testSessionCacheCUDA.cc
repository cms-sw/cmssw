/*
 * Tests for interacting with the SessionCache.
 *
 * Author: Marcel Rieger
 */

#include <stdexcept>
#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "testBaseCUDA.h"

class testSessionCacheCUDA : public testBaseCUDA {
  CPPUNIT_TEST_SUITE(testSessionCacheCUDA);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSessionCacheCUDA);

std::string testSessionCacheCUDA::pyScript() const { return "createconstantgraph.py"; }

void testSessionCacheCUDA::test() {
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

  // load the graph and the session
  std::string pbFile = dataPath_ + "/constantgraph.pb";
  tensorflow::setLogging();
  tensorflow::Options options{backend};

  // load the graph and the session
  tensorflow::SessionCache cache(pbFile, options);

  CPPUNIT_ASSERT(cache.graph.load() != nullptr);
  CPPUNIT_ASSERT(cache.session.load() != nullptr);

  // get a const session pointer
  const tensorflow::Session* session = cache.getSession();
  CPPUNIT_ASSERT(session != nullptr);

  // cleanup
  cache.closeSession();
  CPPUNIT_ASSERT(cache.graph.load() == nullptr);
  CPPUNIT_ASSERT(cache.session.load() == nullptr);
}
