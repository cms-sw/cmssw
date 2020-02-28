/**
   \file
   Test suit for DTDigis

   \author Stefano ARGIRO
   \date 29 Jun 2005

   \note This test is not exaustive     
*/

#include <cppunit/extensions/HelperMacros.h>
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>

using namespace std;

class testDTDigis : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDTDigis);

  CPPUNIT_TEST(testDigiCollectionInsert);
  CPPUNIT_TEST(testDigiCollectionPut);
  CPPUNIT_TEST(testTime2TDCConversion);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void testDigiCollectionInsert();
  void testDigiCollectionPut();
  void testTime2TDCConversion();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testDTDigis);

void testDTDigis::testDigiCollectionPut() {
  DTLayerId layer(2, 3, 8, 1, 4);

  DTDigiCollection digiCollection;

  std::vector<DTDigi> digivec;
  for (int i = 0; i < 10; ++i) {
    DTDigi digi(1 + i, 5 + i, 100 + i);
    digivec.push_back(digi);
  }

  digiCollection.put(std::make_pair(digivec.begin(), digivec.end()), layer);

  // Loop over the DetUnits with digis
  DTDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt = digiCollection.begin(); detUnitIt != digiCollection.end(); ++detUnitIt) {
    const DTLayerId& id = (*detUnitIt).first;
    const DTDigiCollection::Range& range = (*detUnitIt).second;

    //     // We have inserted digis for only one DetUnit...
    CPPUNIT_ASSERT(id == layer);

    // Loop over the digis of this DetUnit
    int i = 0;
    for (DTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second;
         // 	   detUnitIt->second.first;
         // 	 digiIt!=detUnitIt->second.second;
         ++digiIt) {
      CPPUNIT_ASSERT((*digiIt).wire() == 1 + i);
      CPPUNIT_ASSERT((*digiIt).countsTDC() == 5 + i);
      CPPUNIT_ASSERT((*digiIt).number() == 100 + i);
      i++;

      // Test the channel() functionality
      //       DTDigi digi2((*digiIt).channel(),(*digiIt).countsTDC());
      //       CPPUNIT_ASSERT((*digiIt)==digi2);

    }  // for digis in layer
  }    // for layers
}

void testDTDigis::testDigiCollectionInsert() {
  DTDigi digi(1, 5, 4);

  DTLayerId layer(2, 3, 8, 1, 4);

  DTDigiCollection digiCollection;

  digiCollection.insertDigi(layer, digi);

  unsigned int count = 0;

  // Loop over the DetUnits with digis
  DTDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt = digiCollection.begin(); detUnitIt != digiCollection.end(); ++detUnitIt) {
    const DTLayerId& id = (*detUnitIt).first;
    const DTDigiCollection::Range& range = (*detUnitIt).second;

    // We have inserted digis for only one DetUnit...
    CPPUNIT_ASSERT(id == layer);

    // Loop over the digis of this DetUnit
    for (DTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt) {
      //std::cout << (*digiIt) << std::endl;
      CPPUNIT_ASSERT((*digiIt).wire() == 1);
      CPPUNIT_ASSERT((*digiIt).number() == 4);
      CPPUNIT_ASSERT((*digiIt).countsTDC() == 5);

      count++;
    }  // for digis in layer
  }    // for layers

  CPPUNIT_ASSERT(count != 0);
}

void testDTDigis::testTime2TDCConversion() {
  float time = 243;
  float reso = 25. / 32.;
  int tdc = int(time / reso);
  int pos = 2;
  int wire = 1;

  DTDigi digi(wire, time, pos);
  CPPUNIT_ASSERT(digi.countsTDC() == tdc);
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
