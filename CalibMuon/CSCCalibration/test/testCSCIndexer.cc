/**
   \file
   test unit for CSCIndexers (both CSCIndexerStartup and CSCIndexerPostls1)

   What's tested:
    - full round-trip index conversion test for the full ranges of all the indices
    - for the ganged case check the equivalence of ME1a input to ME1a WRT the calculation of various indices
    - full comparison of various indices calculation against the old code

   The old indexers with some code adaptations are saved in CalibMuon/CSCCalibration/test/old_indexers/

   \author Vadim Khotilovich
   \version $Id:$
   \date 2 Dec 2012
*/

static const char CVSId[] = "$Id:$";

#include <cppunit/extensions/HelperMacros.h>
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"

#include "FWCore/Utilities/interface/Exception.h"
#include "CalibMuon/CSCCalibration/interface/CSCIndexerStartup.h"
#include "CalibMuon/CSCCalibration/interface/CSCIndexerPostls1.h"
#include "CalibMuon/CSCCalibration/test/old_indexers/CSCIndexerOldStartup.h"
#include "CalibMuon/CSCCalibration/test/old_indexers/CSCIndexerOldPostls1.h"

#include <iostream>

using namespace std;

typedef CSCIndexerBase::IndexType IndexType;
typedef CSCIndexerBase::LongIndexType LongIndexType;

// an "external polymolrphism" sort of adapter pattern for making unrelated indexer classes polymorphic
class IndexerAdapterBase
{
public:
  virtual ~IndexerAdapterBase() {}
  virtual int ringsInStation(int s) = 0;
  virtual int chambersInRingOfStation(int s, int r) = 0;
  virtual int stripChannelsPerLayer(int s, int r) = 0;
  virtual int chipsPerLayer(int s, int r) = 0;
  virtual int sectorsPerLayer(int s, int r) = 0;
  virtual int chamberIndex(CSCDetId &id) = 0;
  virtual IndexType layerIndex(CSCDetId &id) = 0;
  virtual LongIndexType stripChannelIndex(CSCDetId &id, int strip) = 0;
  virtual IndexType chipIndex(CSCDetId &id, int chip) = 0;
  virtual IndexType gasGainIndex(int hvseg, int chit, CSCDetId &id) = 0;
};

template <class INDEXER>
class IndexerAdapter: public IndexerAdapterBase
{
  INDEXER indexer;
public:
  int ringsInStation(int s) { return indexer.ringsInStation(s); }
  int chambersInRingOfStation(int s, int r) { return indexer.chambersInRingOfStation(s,r); }
  int stripChannelsPerLayer(int s, int r) { return indexer.stripChannelsPerLayer(s,r); }
  int chipsPerLayer(int s, int r) { return indexer.chipsPerLayer(s,r); }
  int sectorsPerLayer(int s, int r) { return indexer.sectorsPerLayer(s,r); }
  int chamberIndex(CSCDetId &id) { return indexer.chamberIndex(id); }
  IndexType layerIndex(CSCDetId &id) { return indexer.layerIndex(id); }
  LongIndexType stripChannelIndex(CSCDetId &id, int strip) { return indexer.stripChannelIndex(id, strip); }
  IndexType chipIndex(CSCDetId &id, int chip) { return indexer.chipIndex(id, chip); }
  IndexType gasGainIndex(int hvseg, int chip, CSCDetId &id) { return indexer.gasGainIndex(hvseg, chip, id); }
};


class testCSCIndexer: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testCSCIndexer);
  
  CPPUNIT_TEST(testStartupChamber);
  CPPUNIT_TEST(testStartupLayer);
  CPPUNIT_TEST(testStartupStripChannel);
  CPPUNIT_TEST(testStartupChip);
  CPPUNIT_TEST(testStartupGasGain);

  CPPUNIT_TEST(testPostls1Chamber);
  CPPUNIT_TEST(testPostls1Layer);
  CPPUNIT_TEST(testPostls1StripChannel);
  CPPUNIT_TEST(testPostls1Chip);
  CPPUNIT_TEST(testPostls1GasGain);

  CPPUNIT_TEST(testStartupChamberME1a);
  CPPUNIT_TEST(testStartupLayerME1a);
  CPPUNIT_TEST(testStartupStripChannelME1a);
  CPPUNIT_TEST(testStartupChipME1a);
  CPPUNIT_TEST(testStartupGasGainME1a);

  CPPUNIT_TEST(testStartupAgainstOldCode);
  CPPUNIT_TEST(testPostls1AgainstOldCode);

  //CPPUNIT_TEST(testFail);

  CPPUNIT_TEST_SUITE_END();

  CSCIndexerBase *indexer_;
  CSCIndexerStartup *indexer_startup_;
  CSCIndexerPostls1 *indexer_postls1_;

  IndexerAdapterBase *indexer_old_;
  IndexerAdapter<CSCIndexerOldStartup> *indexer_old_startup_;
  IndexerAdapter<CSCIndexerOldPostls1> *indexer_old_postls1_;

  void modeStartup() { indexer_ = indexer_startup_; indexer_old_ = indexer_old_startup_; }
  void modePostls1() { indexer_ = indexer_postls1_; indexer_old_ = indexer_old_postls1_; }

public:

  void setUp();
  void tearDown();

  void testChamber();
  void testLayer();
  void testStripChannel();
  void testChip();
  void testGasGain();
  //void testFail();

  void testStartupChamber()      { modeStartup(); testChamber(); }
  void testStartupLayer()        { modeStartup(); testLayer(); }
  void testStartupStripChannel() { modeStartup(); testStripChannel(); }
  void testStartupChip()         { modeStartup(); testChip(); }
  void testStartupGasGain()      { modeStartup(); testGasGain(); }

  void testPostls1Chamber()      { modePostls1(); testChamber(); }
  void testPostls1Layer()        { modePostls1(); testLayer(); }
  void testPostls1StripChannel() { modePostls1(); testStripChannel(); }
  void testPostls1Chip()         { modePostls1(); testChip(); }
  void testPostls1GasGain()      { modePostls1(); testGasGain(); }

  void testStartupChamberME1a();
  void testStartupLayerME1a();
  void testStartupStripChannelME1a();
  void testStartupChipME1a();
  void testStartupGasGainME1a();

  void testStartupAgainstOldCode() { modeStartup(); testAgainstOldCode(); }
  void testPostls1AgainstOldCode() { modePostls1(); testAgainstOldCode(); }

  void testAgainstOldCode();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCSCIndexer);


void testCSCIndexer::setUp()
{
  indexer_ = nullptr;
  indexer_startup_ = new CSCIndexerStartup();
  indexer_postls1_ = new CSCIndexerPostls1();

  indexer_old_ = nullptr;
  indexer_old_startup_ = new IndexerAdapter<CSCIndexerOldStartup>();
  indexer_old_postls1_ = new IndexerAdapter<CSCIndexerOldPostls1>();
}


void testCSCIndexer::tearDown()
{
  delete indexer_startup_;
  delete indexer_postls1_;
  delete indexer_old_startup_;
  delete indexer_old_postls1_;
}


void testCSCIndexer::testChamber()
{
  //std::cout << "\ntestCSCIndexer: testChamber starting... " << std::endl;

  for (LongIndexType i = 1; i <= indexer_->maxChamberIndex(); ++i)
  {
    CSCDetId id = indexer_->detIdFromChamberIndex(i);
    int ie = id.endcap();
    int is = id.station();
    int ir = id.ring();
    int ic = id.chamber();
    IndexType ii = indexer_->chamberIndex(id);

    if (i != ii) cout<<" BAD CHAMBER INDEX: "<<i<<" != "<<ii<<" \t   ("<<ie<<" "<<is<<" "<<ir<<" "<<ic<<")"<<endl;
    CPPUNIT_ASSERT(i == ii);   // loop-back index test
    CPPUNIT_ASSERT(ie >= 1 && ie <= 2);
    CPPUNIT_ASSERT(is >= 1 && is <= 4);
    CPPUNIT_ASSERT(ir >= 1 && ir <= indexer_->offlineRingsInStation(is));
    CPPUNIT_ASSERT(ic >= 1 && ic <= indexer_->chambersInRingOfStation(is, ir));
  }
}


void testCSCIndexer::testLayer()
{
  //std::cout << "\ntestCSCIndexer: testLayer starting... " << std::endl;

  for (LongIndexType i = 1; i <= indexer_->maxLayerIndex(); ++i)
  {
    CSCDetId id = indexer_->detIdFromLayerIndex(i);
    int ie = id.endcap();
    int is = id.station();
    int ir = id.ring();
    int ic = id.chamber();
    int il = id.layer();
    IndexType ii = indexer_->layerIndex(id);

    if (i != ii) cout<<" BAD LAYER INDEX: "<<i<<" != "<<ii<<" \t   ("<<ie<<" "<<is<<" "<<ir<<" "<<ic<<" "<<il<<")"<<endl;
    CPPUNIT_ASSERT(i == ii);   // loop-back index test
    CPPUNIT_ASSERT(ie >= 1 && ie <= 2);
    CPPUNIT_ASSERT(is >= 1 && is <= 4);
    CPPUNIT_ASSERT(ir >= 1 && ir <= indexer_->offlineRingsInStation(is));
    CPPUNIT_ASSERT(ic >= 1 && ic <= indexer_->chambersInRingOfStation(is, ir));
    CPPUNIT_ASSERT(il >= 1 && il <= 6);
  }
}


void testCSCIndexer::testStripChannel()
{
  //std::cout << "\ntestCSCIndexer: testStripChannel starting... " << std::endl;

  for (LongIndexType i = 1; i <= indexer_->maxStripChannelIndex(); ++i)
  {
    std::pair<CSCDetId, CSCIndexerBase::IndexType> t = indexer_->detIdFromStripChannelIndex(i);
    CSCDetId id = t.first;
    int ie = id.endcap();
    int is = id.station();
    int ir = id.ring();
    int ic = id.chamber();
    int il = id.layer();
    int st = t.second;
    LongIndexType ii = indexer_->stripChannelIndex(id, st);

    if (i != ii) cout<<" BAD STRIPCHANNEL INDEX: "<<i<<" != "<<ii<<" \t   ("<<ie<<" "<<is<<" "<<ir<<" "<<ic<<" "<<il<<") "<<st<<endl;
    CPPUNIT_ASSERT(i == ii);   // loop-back index test
    CPPUNIT_ASSERT(ie >= 1 && ie <= 2);
    CPPUNIT_ASSERT(is >= 1 && is <= 4);
    CPPUNIT_ASSERT(ir >= 1 && ir <= indexer_->offlineRingsInStation(is));
    CPPUNIT_ASSERT(ic >= 1 && ic <= indexer_->chambersInRingOfStation(is, ir));
    CPPUNIT_ASSERT(il >= 1 && il <= 6);
    CPPUNIT_ASSERT(st >= 1 && st <= indexer_->stripChannelsPerLayer(is, ir));
  }
}


void testCSCIndexer::testChip()
{
  //std::cout << "\ntestCSCIndexer: testChip starting... " << std::endl;

  for (IndexType i = 1; i <= indexer_->maxChipIndex(); ++i)
  {
    std::pair<CSCDetId, IndexType> t = indexer_->detIdFromChipIndex(i);
    CSCDetId id = t.first;
    int ie = id.endcap();
    int is = id.station();
    int ir = id.ring();
    int ic = id.chamber();
    int il = id.layer();
    int ch = t.second;
    IndexType ii = indexer_->chipIndex(id, ch);

    if (i != ii) cout<<" BAD CHIP INDEX: "<<i<<" != "<<ii<<" \t   ("<<ie<<" "<<is<<" "<<ir<<" "<<ic<<" "<<il<<") "<<ch<<endl;
    CPPUNIT_ASSERT(i == ii);   // loop-back index test
    CPPUNIT_ASSERT(ie >= 1 && ie <= 2);
    CPPUNIT_ASSERT(is >= 1 && is <= 4);
    CPPUNIT_ASSERT(ir >= 1 && ir <= indexer_->offlineRingsInStation(is));
    CPPUNIT_ASSERT(ic >= 1 && ic <= indexer_->chambersInRingOfStation(is, ir));
    CPPUNIT_ASSERT(il >= 1 && il <= 6);
    CPPUNIT_ASSERT(ch >= 1 && ch <= indexer_->chipsPerLayer(is, ir));
  }
}


void testCSCIndexer::testGasGain()
{
  //std::cout << "\ntestCSCIndexer: testGasGain starting... " << std::endl;

  for (IndexType i = 1; i <= indexer_->maxGasGainIndex() ; ++i)
  {
    CSCIndexerBase::GasGainIndexType t = indexer_->detIdFromGasGainIndex(i);
    CSCDetId id = t.get<0>();
    int ie = id.endcap();
    int is = id.station();
    int ir = id.ring();
    int ic = id.chamber();
    int il = id.layer();
    int hv = t.get<1>();
    int ch = t.get<2>();
    IndexType ii = indexer_->gasGainIndex(hv, ch, id);

    if (i != ii) cout<<" BAD GASGAIN INDEX: "<<i<<" != "<<ii<<" \t   ("<<ie<<" "<<is<<" "<<ir<<" "<<ic<<" "<<il<<") "<<hv<<" "<<ch<<endl;
    CPPUNIT_ASSERT(i == ii);   // loop-back index test
    CPPUNIT_ASSERT(ie >= 1 && ie <= 2);
    CPPUNIT_ASSERT(is >= 1 && is <= 4);
    CPPUNIT_ASSERT(ir >= 1 && ir <= indexer_->offlineRingsInStation(is));
    CPPUNIT_ASSERT(ic >= 1 && ic <= indexer_->chambersInRingOfStation(is, ir));
    CPPUNIT_ASSERT(il >= 1 && il <= 6);
    CPPUNIT_ASSERT(hv >= 1 && hv <= indexer_->hvSegmentsPerLayer(is, ir));
    CPPUNIT_ASSERT(ch >= 1 && ch <= indexer_->chipsPerLayer(is, ir));
  }
}


void testCSCIndexer::testStartupChamberME1a()
{
  modeStartup();
  const CSCDetId id_me1a(1,1,4,1);
  const CSCDetId id_me1b(1,1,1,1);
  CPPUNIT_ASSERT(indexer_->chamberIndex(id_me1a) == indexer_->chamberIndex(id_me1b) );
}


void testCSCIndexer::testStartupLayerME1a()
{
  modeStartup();
  const CSCDetId id_me1a(1,1,4,1,2);
  const CSCDetId id_me1b(1,1,1,1,2);
  CPPUNIT_ASSERT(indexer_->layerIndex(id_me1a) == indexer_->layerIndex(id_me1b) );
}


void testCSCIndexer::testStartupStripChannelME1a()
{
  modeStartup();
  const CSCDetId id_me1a(1,1,4,1,2);
  const CSCDetId id_me1b(1,1,1,1,2);
  const IndexType istrip = 66;
  CPPUNIT_ASSERT(indexer_->stripChannelIndex(id_me1a, istrip) == indexer_->stripChannelIndex(id_me1b, istrip) );
}


void testCSCIndexer::testStartupChipME1a()
{
  modeStartup();
  const CSCDetId id_me1a(1,1,4,1,2);
  const CSCDetId id_me1b(1,1,1,1,2);
  const IndexType ichip = 5;
  CPPUNIT_ASSERT(indexer_->chipIndex(id_me1a, ichip) == indexer_->chipIndex(id_me1b, ichip) );
}


void testCSCIndexer::testStartupGasGainME1a()
{
  modeStartup();
  const CSCDetId id_me1a(1,1,4,1,2);
  const CSCDetId id_me1b(1,1,1,1,2);
  const IndexType istrip = 66;
  const IndexType iwire = 4;
  CPPUNIT_ASSERT(indexer_->gasGainIndex(id_me1a, istrip, iwire) == indexer_->gasGainIndex(id_me1b, istrip, iwire) );
}


void testCSCIndexer::testAgainstOldCode()
{
  for (int e = 1; e <= 2; ++e)
    for (int s = 1; s <= 4; ++s)
    {
      int rmax = indexer_->ringsInStation(s);
      CPPUNIT_ASSERT(rmax == indexer_old_->ringsInStation(s) );

      if (s == 1 && indexer_->name() == "CSCIndexerPostls1") rmax = 4;

      for ( int r = 1; r <= rmax; ++r )
      {
        int cmax = indexer_->chambersInRingOfStation(s, r);
        CPPUNIT_ASSERT(cmax == indexer_old_->chambersInRingOfStation(s, r) );

        CPPUNIT_ASSERT(indexer_->stripChannelsPerLayer(s, r) ==
                       indexer_old_->stripChannelsPerLayer(s, r) );
        int stripmax = indexer_->stripChannelsPerOnlineLayer(s, r);

        CPPUNIT_ASSERT(indexer_->chipsPerLayer(s, r) ==
                       indexer_old_->chipsPerLayer(s, r) );
        int chipmax = indexer_->chipsPerOnlineLayer(s, r);

        CPPUNIT_ASSERT(indexer_->sectorsPerLayer(s, r) == indexer_old_->sectorsPerLayer(s, r) );
        int hvsegmax = indexer_->sectorsPerOnlineLayer(s, r);

        for ( int c = 1; c <= cmax; ++c )
        {
          CSCDetId cid(e,s,r,c);
          CPPUNIT_ASSERT( indexer_->chamberIndex(cid) == indexer_old_->chamberIndex(cid));

          for ( int l = 1; l <= 6; ++l )
          {
            CSCDetId id(e,s,r,c,l);
            CPPUNIT_ASSERT( indexer_->layerIndex(id) == indexer_old_->layerIndex(id));

            for (int strip = 1; strip <= stripmax; ++strip)
            {
              CPPUNIT_ASSERT( indexer_->stripChannelIndex(id, strip) == indexer_old_->stripChannelIndex(id, strip));
            }

            for (int chip = 1; chip <= chipmax; ++chip)
            {
              CPPUNIT_ASSERT( indexer_->chipIndex(id, chip) == indexer_old_->chipIndex(id, chip));

              for (int hvseg = 1; hvseg <= hvsegmax; ++hvseg)
              {
                CPPUNIT_ASSERT( indexer_->gasGainIndex(hvseg, chip, id) == indexer_old_->gasGainIndex(hvseg, chip, id));
              }
            }
          }
        }
      }
    }
}

/*
void testCSCIndexer::testFail(){
  
  // std::cout << "\ntestCSCIndexer: testFail starting... " << std::endl;

  // construct using an invalid input index
  try {
    // Invalid layer
    CSCDetId detid(3,1,1,1,7);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0); 
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    //    std::cout << "\ntestCSCDetId: testFail exception caught " << std::endl;
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }
  
}
*/
