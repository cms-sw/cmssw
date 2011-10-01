#include "RecoTauTag/RecoTau/interface/CombinatoricGenerator.h"
#include <cppunit/extensions/HelperMacros.h>
#include <sstream>
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <iostream>

typedef std::vector<int> vint;

// a list of the combo + remaining items
typedef std::pair<vint, vint> ComboRemainder;

// a list of the combo + remaining items
typedef std::vector<ComboRemainder> VComboRemainder;

template<typename G>
VComboRemainder getAllCombinations(G& generator) {
  VComboRemainder output;
  typedef typename G::iterator iterator;
  typedef typename G::combo_iterator combo_iterator;
  for (iterator combo = generator.begin(); combo != generator.end();
      ++combo) {
    ComboRemainder thisCombo;
    for (combo_iterator item = combo->combo_begin();
        item != combo->combo_end(); ++item) {
      thisCombo.first.push_back(*item);
    }

    for (combo_iterator item = combo->remainder_begin();
        item != combo->remainder_end(); ++item) {
      thisCombo.second.push_back(*item);
    }

    output.push_back(thisCombo);
  }
  return output;
}

class testCombGenerator : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testCombGenerator);
  CPPUNIT_TEST(testEmptyCollection);
  CPPUNIT_TEST(testLessThanDesiredCollection);
  CPPUNIT_TEST(testNormalBehavior);
  CPPUNIT_TEST_SUITE_END();

  public:

     void setup() {}
     typedef reco::tau::CombinatoricGenerator<vint> Generator;

     void testNormalBehavior() {
       std::cout << "Testing normal behavior" << std::endl;
       vint toCombize;
       toCombize.push_back(1);
       toCombize.push_back(2);
       toCombize.push_back(3);
       toCombize.push_back(4);

       Generator pairGenerator(toCombize.begin(), toCombize.end(), 2);

       VComboRemainder pairResults = getAllCombinations(pairGenerator);

       // 4 choose 2 = 6

       CPPUNIT_ASSERT(pairResults.size() == 6);

       ComboRemainder firstExpResult;
       firstExpResult.first.push_back(1);
       firstExpResult.first.push_back(2);
       firstExpResult.second.push_back(3);
       firstExpResult.second.push_back(4);

       CPPUNIT_ASSERT(firstExpResult == pairResults[0]);

       ComboRemainder secondExpResult;
       secondExpResult.first.push_back(1);
       secondExpResult.first.push_back(3);
       secondExpResult.second.push_back(2);
       secondExpResult.second.push_back(4);

       CPPUNIT_ASSERT(secondExpResult == pairResults[1]);

       ComboRemainder lastExpResult;
       lastExpResult.first.push_back(3);
       lastExpResult.first.push_back(4);
       lastExpResult.second.push_back(1);
       lastExpResult.second.push_back(2);

       Generator tripletGen(toCombize.begin(), toCombize.end(), 3);

       VComboRemainder tripletResults = getAllCombinations(tripletGen);
       CPPUNIT_ASSERT(tripletResults.size() == 4);

       // This one should have only one combinatoric
       Generator quadGen(toCombize.begin(), toCombize.end(), 4);
       VComboRemainder quadResults = getAllCombinations(quadGen);
       CPPUNIT_ASSERT(quadResults.size() == 1);
       // All should be in the combinatoric, none in the remainder
       CPPUNIT_ASSERT(quadResults[0].first.size() == 4);
       CPPUNIT_ASSERT(quadResults[0].second.size() == 0);
     };

     void testLessThanDesiredCollection() {
       std::cout << "Testing less than behavior" << std::endl;
       vint toCombize;
       toCombize.push_back(1);
       toCombize.push_back(2);

       Generator tripletGen(toCombize.begin(), toCombize.end(), 3);
       VComboRemainder tripletResults = getAllCombinations(tripletGen);
       // Can't make any combos - 2 choose 3 doesn't make sense.
       CPPUNIT_ASSERT(tripletResults.size() == 0);
     }

     void testEmptyCollection() {
       std::cout << "Testing empty collection" << std::endl;
       // A requested empty collection should return 1 empty combo
       vint toCombize;
       toCombize.push_back(1);
       toCombize.push_back(2);
       toCombize.push_back(3);
       toCombize.push_back(4);

       Generator zeroGen(toCombize.begin(), toCombize.end(), 0);
       VComboRemainder zeroResults = getAllCombinations(zeroGen);
       CPPUNIT_ASSERT(zeroResults.size() == 1);
       CPPUNIT_ASSERT(zeroResults[0].first.size() == 0);
       CPPUNIT_ASSERT(zeroResults[0].second.size() == 4);
     }

};

CPPUNIT_TEST_SUITE_REGISTRATION(testCombGenerator);
