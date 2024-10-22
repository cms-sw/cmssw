#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/TCDS/interface/BSTRecord.h"
#include "DataFormats/TCDS/interface/L1aInfo.h"
#include "DataFormats/TCDS/interface/TCDSRaw.h"
#include "DataFormats/TCDS/interface/TCDSRecord.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

class TestTCDSRecord : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestTCDSRecord);
  CPPUNIT_TEST(testHeader);
  CPPUNIT_TEST(testBstRecord);
  CPPUNIT_TEST(testL1aHistory);
  CPPUNIT_TEST(testBgoHistory);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override {}

  void testHeader();
  void testBstRecord();
  void testL1aHistory();
  void testBgoHistory();

private:
  const std::string dumpFileName = "dump_run302403_event00112245_fed1024.txt";
  std::vector<uint32_t> data;
  TCDSRecord tcdsRecord;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestTCDSRecord);

void TestTCDSRecord::setUp() {
  if (data.empty()) {
    char* cmsswBase;
    cmsswBase = getenv("CMSSW_BASE");
    std::ostringstream dumpFileStr;
    dumpFileStr << cmsswBase << "/src/DataFormats/TCDS/test/" << dumpFileName;

    std::ifstream dumpFile(dumpFileStr.str().c_str());
    uint32_t address;

    std::string line, column;
    while (dumpFile.good()) {
      getline(dumpFile, line);
      //std::cout << line << std::endl;
      std::istringstream iss(line);
      if (!(iss >> std::hex >> address >> column))
        continue;
      for (int i = 0; i < 4; ++i) {
        uint32_t value;
        if (iss >> std::hex >> value) {
          data.push_back(value);
        }
      }
    }
    dumpFile.close();

    CPPUNIT_ASSERT(data.size() == 238);

    tcdsRecord = TCDSRecord((unsigned char*)data.data());
    //std::cout << tcdsRecord << std::endl;
  }
}

void TestTCDSRecord::testHeader() {
  CPPUNIT_ASSERT(tcdsRecord.getMacAddress() == 0x80030f30044);
  CPPUNIT_ASSERT(tcdsRecord.getSwVersion() == 0x3005002);
  CPPUNIT_ASSERT(tcdsRecord.getFwVersion() == 0x5601222a);
  CPPUNIT_ASSERT(tcdsRecord.getRecordVersion() == 1);
  CPPUNIT_ASSERT(tcdsRecord.getRunNumber() == 302403);
  CPPUNIT_ASSERT(tcdsRecord.getNibble() == 23);
  CPPUNIT_ASSERT(tcdsRecord.getLumiSection() == 6);
  CPPUNIT_ASSERT(tcdsRecord.getEventType() == 2);
  CPPUNIT_ASSERT(tcdsRecord.getInputs() == 0);
  CPPUNIT_ASSERT(tcdsRecord.getOrbitNr() == 1403708);
  CPPUNIT_ASSERT(tcdsRecord.getBXID() == 3490);
  CPPUNIT_ASSERT(tcdsRecord.getTriggerCount() == 112245);
  CPPUNIT_ASSERT(tcdsRecord.getEventNumber() == 112245);
  CPPUNIT_ASSERT(tcdsRecord.getBstReceptionStatus() == TCDSRecord::BSTstatus::Okay);
  CPPUNIT_ASSERT(tcdsRecord.getActivePartitions().none());
}

void TestTCDSRecord::testBstRecord() {
  CPPUNIT_ASSERT(tcdsRecord.getBST().getGpsTime() == 6462610227703219200);
  CPPUNIT_ASSERT(tcdsRecord.getBST().getBstMaster() == 1);
  CPPUNIT_ASSERT(tcdsRecord.getBST().getTurnCount() == 49907439);
  CPPUNIT_ASSERT(tcdsRecord.getBST().getLhcFill() == 6172);
  CPPUNIT_ASSERT(tcdsRecord.getBST().getBeamMode() == BSTRecord::BeamMode::RAMPDOWN);
  CPPUNIT_ASSERT(tcdsRecord.getBST().getParticleBeam1() == BSTRecord::Particle::PROTON);
  CPPUNIT_ASSERT(tcdsRecord.getBST().getParticleBeam2() == BSTRecord::Particle::PROTON);
  CPPUNIT_ASSERT(tcdsRecord.getBST().getBeamMomentum() == 1302);
  CPPUNIT_ASSERT(tcdsRecord.getBST().getIntensityBeam1() == 0);
  CPPUNIT_ASSERT(tcdsRecord.getBST().getIntensityBeam2() == 0);
}

void TestTCDSRecord::testL1aHistory() {
  TCDSRecord::L1aHistory l1aHistory = tcdsRecord.getFullL1aHistory();
  CPPUNIT_ASSERT(l1aHistory.size() == tcds::l1aHistoryDepth_v1);

  uint8_t index = 0;
  const uint64_t expectedOrbits[tcds::l1aHistoryDepth_v1] = {1403697,
                                                             1403691,
                                                             1403686,
                                                             1403685,
                                                             1403678,
                                                             1403675,
                                                             1403671,
                                                             1403664,
                                                             1403660,
                                                             1403651,
                                                             1403650,
                                                             1403643,
                                                             1403632,
                                                             1403630,
                                                             1403620,
                                                             1403616};
  const uint64_t expectedBXIDs[tcds::l1aHistoryDepth_v1] = {
      297, 2090, 3136, 60, 249, 580, 3375, 2049, 524, 2518, 1083, 2571, 612, 2130, 2089, 3049};
  const uint64_t expectedEventTypes[tcds::l1aHistoryDepth_v1] = {3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 1, 3, 1, 3, 3};

  for (auto l1a : l1aHistory) {
    CPPUNIT_ASSERT(l1a.getIndex() == -1 - index);
    CPPUNIT_ASSERT(l1a.getOrbitNr() == expectedOrbits[index]);
    CPPUNIT_ASSERT(l1a.getBXID() == expectedBXIDs[index]);
    CPPUNIT_ASSERT(l1a.getEventType() == expectedEventTypes[index]);
    ++index;
  }
}

void TestTCDSRecord::testBgoHistory() {
  const uint64_t expectedOrbits[tcds::bgoCount_v1] = {
      1400831, 1403708, 1403708, 0, 0, 4518870, 0, 8, 4518878, 7, 0, 1403708, 0, 1403707, 1403606};

  CPPUNIT_ASSERT(tcdsRecord.getLastOrbitCounter0() == 4518878);
  CPPUNIT_ASSERT(tcdsRecord.getLastTestEnable() == 1403708);
  CPPUNIT_ASSERT(tcdsRecord.getLastResync() == 4518870);
  CPPUNIT_ASSERT(tcdsRecord.getLastStart() == 7);
  CPPUNIT_ASSERT(tcdsRecord.getLastEventCounter0() == 8);
  CPPUNIT_ASSERT(tcdsRecord.getLastHardReset() == 0);

  for (uint8_t bgo = 0; bgo < tcds::bgoCount_v1; ++bgo) {
    CPPUNIT_ASSERT(tcdsRecord.getOrbitOfLastBgo(bgo) == expectedOrbits[bgo]);
  }
}
