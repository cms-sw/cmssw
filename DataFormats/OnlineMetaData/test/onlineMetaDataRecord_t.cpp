#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/OnlineMetaData/interface/CTPPSRecord.h"
#include "DataFormats/OnlineMetaData/interface/DCSRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineMetaDataRaw.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string.h>
#include <vector>

class TestOnlineMetaDataRecord : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestOnlineMetaDataRecord);
  CPPUNIT_TEST(testDCSRecord_v1);
  CPPUNIT_TEST(testDCSRecord_v2);
  CPPUNIT_TEST(testDCSRecord_v3);
  CPPUNIT_TEST(testOnlineLuminosityRecord_v1);
  CPPUNIT_TEST(testOnlineLuminosityRecord_v2);
  CPPUNIT_TEST(testCTPPSRecord_v2);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override{};
  void tearDown() override;

  void testDCSRecord_v1();
  void testDCSRecord_v2();
  void testDCSRecord_v3();
  void testOnlineLuminosityRecord_v1();
  void testOnlineLuminosityRecord_v2();
  void testCTPPSRecord_v2();

private:
  const unsigned char* readPayload(const std::string& dumpFileName);

  std::vector<uint32_t> data;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestOnlineMetaDataRecord);

const unsigned char* TestOnlineMetaDataRecord::readPayload(const std::string& dumpFileName) {
  const std::string CMSSW_BASE(std::getenv("CMSSW_BASE"));
  const std::string CMSSW_RELEASE_BASE(std::getenv("CMSSW_RELEASE_BASE"));
  const std::string dumpFilePath = "/src/DataFormats/OnlineMetaData/test/" + dumpFileName;
  const std::string fullPath = std::filesystem::exists((CMSSW_BASE + dumpFilePath).c_str())
                                   ? CMSSW_BASE + dumpFilePath
                                   : CMSSW_RELEASE_BASE + dumpFilePath;

  std::ifstream dumpFile(fullPath.c_str());
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

  return reinterpret_cast<unsigned char*>(data.data());
}

void TestOnlineMetaDataRecord::tearDown() { data.clear(); }

float castToFloat(uint32_t value) {
  float f;
  memcpy(&f, &value, sizeof(uint32_t));
  return f;
}

void TestOnlineMetaDataRecord::testDCSRecord_v1() {
  const unsigned char* payload = readPayload("dump_run000001_event00057185_fed1022.txt");
  const online::Data_v1* data_v1 = reinterpret_cast<online::Data_v1 const*>(payload + FEDHeader::length);
  DCSRecord dcs(data_v1->dcs);
  std::cout << dcs << std::endl;

  // DIP timestamp is in milliseconds
  const uint64_t ts = dcs.timestamp().unixTime() * 1000UL + dcs.timestamp().microsecondOffset() / 1000;
  CPPUNIT_ASSERT_EQUAL(static_cast<uint64_t>(0x160036025f6), ts);
  CPPUNIT_ASSERT(dcs.highVoltageReady(DCSRecord::Partition::HF));
  CPPUNIT_ASSERT(!dcs.highVoltageReady(DCSRecord::Partition::CASTOR));
  CPPUNIT_ASSERT(dcs.highVoltageReady(DCSRecord::Partition::ESm));
  CPPUNIT_ASSERT_EQUAL(castToFloat(0x468de7eb), dcs.magnetCurrent());
}

void TestOnlineMetaDataRecord::testDCSRecord_v2() {
  const unsigned char* payload = readPayload("dump_run000001_event00013761_fed1022.txt");
  const online::Data_v2* data_v2 = reinterpret_cast<online::Data_v2 const*>(payload + FEDHeader::length);
  DCSRecord dcs(data_v2->dcs);
  std::cout << dcs << std::endl;

  // DIP timestamp is in milliseconds
  const uint64_t ts = dcs.timestamp().unixTime() * 1000UL + dcs.timestamp().microsecondOffset() / 1000;
  CPPUNIT_ASSERT_EQUAL(static_cast<uint64_t>(0x1616b52f433), ts);
  CPPUNIT_ASSERT(dcs.highVoltageReady(DCSRecord::Partition::CSCp));
  CPPUNIT_ASSERT(!dcs.highVoltageReady(DCSRecord::Partition::BPIX));
  CPPUNIT_ASSERT(dcs.highVoltageReady(DCSRecord::Partition::TOB));
  CPPUNIT_ASSERT_EQUAL(castToFloat(0x3ccd2785), dcs.magnetCurrent());
}

void TestOnlineMetaDataRecord::testDCSRecord_v3() {
  const unsigned char* payload = readPayload("dump_run000001_event1350583585_fed1022.txt");
  const online::Data_v3* data_v3 = reinterpret_cast<online::Data_v3 const*>(payload + FEDHeader::length);
  DCSRecord dcs(data_v3->dcs);

  // DIP timestamp is in milliseconds
  const uint64_t ts = dcs.timestamp().unixTime() * 1000UL + dcs.timestamp().microsecondOffset() / 1000;
  CPPUNIT_ASSERT_EQUAL(static_cast<uint64_t>(0x17d6fae1ad6), ts);
  CPPUNIT_ASSERT(dcs.highVoltageValid(DCSRecord::Partition::CSCp));
  CPPUNIT_ASSERT(!dcs.highVoltageReady(DCSRecord::Partition::CSCp));
  CPPUNIT_ASSERT(dcs.highVoltageValid(DCSRecord::Partition::BPIX));
  CPPUNIT_ASSERT(!dcs.highVoltageReady(DCSRecord::Partition::BPIX));
  CPPUNIT_ASSERT(dcs.highVoltageValid(DCSRecord::Partition::TOB));
  CPPUNIT_ASSERT(!dcs.highVoltageReady(DCSRecord::Partition::TOB));
  CPPUNIT_ASSERT(!dcs.highVoltageValid(DCSRecord::Partition::ZDC));
  CPPUNIT_ASSERT(!dcs.highVoltageValid(DCSRecord::Partition::CASTOR));
  CPPUNIT_ASSERT_EQUAL(castToFloat(0x3D69F92E), dcs.magnetCurrent());
}

void TestOnlineMetaDataRecord::testOnlineLuminosityRecord_v1() {
  const unsigned char* payload = readPayload("dump_run000001_event00057185_fed1022.txt");
  const online::Data_v1* data_v1 = reinterpret_cast<online::Data_v1 const*>(payload + FEDHeader::length);
  OnlineLuminosityRecord lumi(data_v1->luminosity);
  std::cout << lumi << std::endl;

  // DIP timestamp is in milliseconds
  const uint64_t ts = lumi.timestamp().unixTime() * 1000UL + lumi.timestamp().microsecondOffset() / 1000;
  CPPUNIT_ASSERT_EQUAL(static_cast<uint64_t>(0x160070979e4), ts);
  CPPUNIT_ASSERT_EQUAL(static_cast<uint16_t>(0x59), lumi.lumiSection());
  CPPUNIT_ASSERT_EQUAL(static_cast<uint16_t>(0x30), lumi.lumiNibble());
  CPPUNIT_ASSERT_EQUAL(static_cast<float>(0), lumi.instLumi());
  CPPUNIT_ASSERT_EQUAL(static_cast<float>(0), lumi.avgPileUp());
}

void TestOnlineMetaDataRecord::testOnlineLuminosityRecord_v2() {
  const unsigned char* payload = readPayload("dump_run000001_event00013761_fed1022.txt");
  const online::Data_v2* data_v2 = reinterpret_cast<online::Data_v2 const*>(payload + FEDHeader::length);
  OnlineLuminosityRecord lumi(data_v2->luminosity);
  std::cout << lumi << std::endl;

  // DIP timestamp is in milliseconds
  const uint64_t ts = lumi.timestamp().unixTime() * 1000UL + lumi.timestamp().microsecondOffset() / 1000;
  CPPUNIT_ASSERT_EQUAL(static_cast<uint64_t>(0x161070979e4), ts);
  CPPUNIT_ASSERT_EQUAL(static_cast<uint16_t>(0x45), lumi.lumiSection());
  CPPUNIT_ASSERT_EQUAL(static_cast<uint16_t>(0x33), lumi.lumiNibble());
  CPPUNIT_ASSERT_EQUAL(static_cast<float>(0), lumi.instLumi());
  CPPUNIT_ASSERT_EQUAL(static_cast<float>(0), lumi.avgPileUp());
}

void TestOnlineMetaDataRecord::testCTPPSRecord_v2() {
  const unsigned char* payload = readPayload("dump_run000001_event6135_fed1022.txt");
  const online::Data_v2* data_v2 = reinterpret_cast<online::Data_v2 const*>(payload + FEDHeader::length);
  CTPPSRecord ctpps(data_v2->ctpps);
  std::cout << ctpps << std::endl;

  // DIP timestamp is in milliseconds
  const uint64_t ts = ctpps.timestamp().unixTime() * 1000UL + ctpps.timestamp().microsecondOffset() / 1000;
  CPPUNIT_ASSERT_EQUAL(static_cast<uint64_t>(0x18799D62E9A), ts);
  CPPUNIT_ASSERT_EQUAL(CTPPSRecord::Status::unused, ctpps.status(CTPPSRecord::RomanPot::RP_45_210_FR_BT));
  CPPUNIT_ASSERT_EQUAL(CTPPSRecord::Status::unused, ctpps.status(CTPPSRecord::RomanPot::RP_45_220_FR_TP));
  CPPUNIT_ASSERT_EQUAL(CTPPSRecord::Status::unused, ctpps.status(CTPPSRecord::RomanPot::RP_45_220_NR_TP));
  CPPUNIT_ASSERT_EQUAL(CTPPSRecord::Status::bad, ctpps.status(CTPPSRecord::RomanPot::RP_56_220_NR_HR));
}
