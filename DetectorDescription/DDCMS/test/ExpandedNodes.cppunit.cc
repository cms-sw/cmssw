#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/DDCMS/interface/ExpandedNodes.h"

#include <iostream>

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

using namespace cms;
using namespace std;

class testExpandedNodes : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testExpandedNodes);
  CPPUNIT_TEST(checkExpandedNodes);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override {}
  void checkExpandedNodes();

private:
  ExpandedNodes nodes_;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testExpandedNodes);

void testExpandedNodes::setUp() {
  nodes_.tags = {1., 2., 3.};
  nodes_.offsets = {1., 2., 3.};
  nodes_.copyNos = {1, 2, 3};
}

void testExpandedNodes::checkExpandedNodes() {
  cout << "Expanded Nodes...\n";
  CPPUNIT_ASSERT(nodes_.tags.size() == nodes_.offsets.size());
  CPPUNIT_ASSERT(nodes_.tags.size() == nodes_.copyNos.size());

  for (auto const& i : nodes_.tags)
    cout << i << " ";
}
