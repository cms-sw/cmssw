/*
 * Tests for interacting with the SessionCache.
 *
 * Author: Marcel Rieger
 */

#include <stdexcept>
#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "testBase.h"

class testSessionCache : public testBase {
  CPPUNIT_TEST_SUITE(testSessionCache);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void checkAll() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSessionCache);

std::string testSessionCache::pyScript() const { return "createconstantgraph.py"; }

void testSessionCache::checkAll() {
  std::string pbFile = dataPath_ + "/constantgraph.pb";

  tensorflow::setLogging();

  // load the graph and the session
  tensorflow::SessionCache cache(pbFile);
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
