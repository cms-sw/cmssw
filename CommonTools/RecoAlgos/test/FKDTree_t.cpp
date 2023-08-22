#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

class TestFKDTree : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestFKDTree);
  CPPUNIT_TEST(test2D);
  CPPUNIT_TEST(test3D);
  CPPUNIT_TEST_SUITE_END();

public:
  /// run all test tokenizer
  void test2D();
  void test3D();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestFKDTree);

#include "CommonTools/RecoAlgos/interface/FKDTree.h"
#include "FWCore/Utilities/interface/Exception.h"

void TestFKDTree::test2D() {
  try {
    FKDTree<float, 2> tree;
    float minX = 0.2;
    float minY = 0.1;
    float maxX = 0.7;
    float maxY = 0.9;
    unsigned int numberOfPointsInTheBox = 1000;
    unsigned int numberOfPointsOutsideTheBox = 5000;
    std::vector<FKDPoint<float, 2> > points;
    std::vector<unsigned int> result;

    FKDPoint<float, 2> minPoint(minX, minY);
    FKDPoint<float, 2> maxPoint(maxX, maxY);
    unsigned int id = 0;

    for (unsigned int i = 0; i < numberOfPointsInTheBox; ++i) {
      float x = minX + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / std::fabs(maxX - minX));

      float y = minY + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / std::fabs(maxY - minY));

      points.emplace_back(x, y, id);
      id++;
    }

    for (unsigned int i = 0; i < numberOfPointsOutsideTheBox; ++i) {
      float x = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
      float y = 1.f;
      if (x <= maxX && x >= minX) {
        y = maxY + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / std::fabs(1.f - maxY));
        if (y == maxY)
          y = 1.f;  // avoid y = maxY
      }
      points.emplace_back(x, y, id);
      id++;
    }
    tree.build(points);
    tree.search(minPoint, maxPoint, result);
    CPPUNIT_ASSERT_EQUAL((unsigned int)tree.size(), numberOfPointsInTheBox + numberOfPointsOutsideTheBox);

    CPPUNIT_ASSERT_EQUAL((unsigned int)result.size(), numberOfPointsInTheBox);
  } catch (cms::Exception& e) {
    std::cerr << e.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}
void TestFKDTree::test3D() {
  try {
    FKDTree<float, 3> tree;
    float minX = 0.2;
    float minY = 0.1;
    float minZ = 0.1;
    float maxX = 0.7;
    float maxY = 0.9;
    float maxZ = 0.3;

    unsigned int numberOfPointsInTheBox = 1000;
    unsigned int numberOfPointsOutsideTheBox = 5000;
    std::vector<FKDPoint<float, 3> > points;
    std::vector<unsigned int> result;

    FKDPoint<float, 3> minPoint(minX, minY, minZ);
    FKDPoint<float, 3> maxPoint(maxX, maxY, maxZ);
    unsigned int id = 0;

    for (unsigned int i = 0; i < numberOfPointsInTheBox; ++i) {
      float x = minX + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / std::fabs(maxX - minX));

      float y = minY + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / std::fabs(maxY - minY));
      float z = minZ + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / std::fabs(maxZ - minZ));

      points.emplace_back(x, y, z, id);
      id++;
    }

    for (unsigned int i = 0; i < numberOfPointsOutsideTheBox; ++i) {
      float x = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
      float y;
      if (x <= maxX && x >= minX) {
        y = maxY + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / std::fabs(1.f - maxY));
        if (y == maxY)
          y = 1.f;  // avoid y = maxY
      }
      float z = minZ + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / std::fabs(maxZ - minZ));

      points.emplace_back(x, y, z, id);
      id++;
    }
    tree.build(points);
    tree.search(minPoint, maxPoint, result);
    CPPUNIT_ASSERT_EQUAL((unsigned int)tree.size(), numberOfPointsInTheBox + numberOfPointsOutsideTheBox);
    CPPUNIT_ASSERT_EQUAL((unsigned int)result.size(), numberOfPointsInTheBox);

  } catch (cms::Exception& e) {
    std::cerr << e.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}
