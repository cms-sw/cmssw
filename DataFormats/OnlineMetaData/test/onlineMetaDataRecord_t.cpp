#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/OnlineMetaData/interface/DCSRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineBeamSpotRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineMetaDataRaw.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string.h>
#include <vector>

#include <boost/filesystem.hpp>


class TestOnlineMetaDataRecord: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestOnlineMetaDataRecord);
  CPPUNIT_TEST(testDCSRecord);
  CPPUNIT_TEST(testOnlineBeamSpotRecord);
  CPPUNIT_TEST(testOnlineLuminosityRecord);
  CPPUNIT_TEST_SUITE_END();

public:

  void setUp() override;
  void tearDown() override {}

  void testDCSRecord();
  void testOnlineBeamSpotRecord();
  void testOnlineLuminosityRecord();

private:
  const std::string dumpFileName = "dump_run000001_event00057185_fed1022.txt";
  std::vector<uint32_t> data;
  online::Data_v1 const* onlineMetaData;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestOnlineMetaDataRecord);


void TestOnlineMetaDataRecord::setUp() {

  if ( data.empty() ) {
    std::string CMSSW_BASE(std::getenv("CMSSW_BASE"));
    std::string CMSSW_RELEASE_BASE(std::getenv("CMSSW_RELEASE_BASE"));
    std::string dumpFileName("/src/DataFormats/OnlineMetaData/test/dump_run000001_event00057185_fed1022.txt");
    std::string fullPath = boost::filesystem::exists((CMSSW_BASE+dumpFileName).c_str()) ? CMSSW_BASE+dumpFileName : CMSSW_RELEASE_BASE+dumpFileName;

    std::ifstream dumpFile(fullPath.c_str());
    uint32_t address;

    std::string line, column;
    while ( dumpFile.good() ) {
      getline(dumpFile, line);
      //std::cout << line << std::endl;
      std::istringstream iss(line);
      if ( ! (iss >> std::hex >> address >> column) ) continue;
      for ( int i = 0; i < 4; ++i) {
        uint32_t value;
        if ( iss >> std::hex >> value ) {
          data.push_back(value);
        }
      }
    }
    dumpFile.close();

    CPPUNIT_ASSERT_EQUAL( static_cast<size_t>(34),data.size() );

    onlineMetaData = reinterpret_cast<online::Data_v1 const*>(data.data());
  }
}


float castToFloat(uint32_t value) {
  float f;
  memcpy(&f,&value,sizeof(uint32_t));
  return f;
}


void TestOnlineMetaDataRecord::testDCSRecord() {

  DCSRecord dcs(onlineMetaData->dcs);
  std::cout << dcs << std::endl;

  // DIP timestamp is in milliseconds
  const uint64_t ts = dcs.timestamp().unixTime() * 1000UL + dcs.timestamp().microsecondOffset()/1000;
  CPPUNIT_ASSERT_EQUAL( static_cast<uint64_t>(0x160036025f6),ts );
  CPPUNIT_ASSERT( dcs.highVoltageReady(DCSRecord::Partition::HF) );
  CPPUNIT_ASSERT( ! dcs.highVoltageReady(DCSRecord::Partition::CASTOR) );
  CPPUNIT_ASSERT( dcs.highVoltageReady(DCSRecord::Partition::ESm) );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x468de7eb),dcs.magnetCurrent() );
}


void TestOnlineMetaDataRecord::testOnlineBeamSpotRecord() {

  OnlineBeamSpotRecord beamSpot(onlineMetaData->beamSpot);
  std::cout << beamSpot << std::endl;

  // DIP timestamp is in milliseconds
  const uint64_t ts = beamSpot.timestamp().unixTime() * 1000UL + beamSpot.timestamp().microsecondOffset()/1000;
  CPPUNIT_ASSERT_EQUAL( static_cast<uint64_t>(0x15ff7e03190),ts );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x3da6a162),beamSpot.x() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0xbcf2b488),beamSpot.y() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x3edb7ed4),beamSpot.z() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x38bc750c),beamSpot.dxdz() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0xb84079b1),beamSpot.dydz() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x3d6fb475),beamSpot.errX() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x3866b990),beamSpot.errY() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x00000000),beamSpot.errZ() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x37841abe),beamSpot.errDxdz() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x37814f2b),beamSpot.errDydz() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x3ae1af07),beamSpot.widthX() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x3ada5b00),beamSpot.widthY() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x405c8049),beamSpot.sigmaZ() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x387463de),beamSpot.errWidthX() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x387463de),beamSpot.errWidthY() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x3d29804d),beamSpot.errSigmaZ() );
}


void TestOnlineMetaDataRecord::testOnlineLuminosityRecord() {

  OnlineLuminosityRecord lumi(onlineMetaData->luminosity);
  std::cout << lumi << std::endl;

  // DIP timestamp is in milliseconds
  const uint64_t ts = lumi.timestamp().unixTime() * 1000UL + lumi.timestamp().microsecondOffset()/1000;
  CPPUNIT_ASSERT_EQUAL( static_cast<uint64_t>(0x160070979e4),ts );
  CPPUNIT_ASSERT_EQUAL( static_cast<uint16_t>(0x59),lumi.lumiSection() );
  CPPUNIT_ASSERT_EQUAL( static_cast<uint16_t>(0x30),lumi.lumiNibble() );
  CPPUNIT_ASSERT_EQUAL( static_cast<float>(0),lumi.instLumi() );
  CPPUNIT_ASSERT_EQUAL( static_cast<float>(0),lumi.avgPileUp() );
}
