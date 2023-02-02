/*
 * Tests for interacting with the SessionCache.
 *
 * Author: Marcel Rieger
 */

#include <stdexcept>
#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "testBase.h"

class testSessionCacheCUDA : public testBase {
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
  std::string pbFile = dataPath_ + "/constantgraph.pb";

  std::cout << "Testing CUDA backend" << std::endl;
  tensorflow::Backend backend = tensorflow::Backend::cuda;

  tensorflow::setLogging();

  // load the graph and the session
  tensorflow::SessionCache cache(pbFile, backend);
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
