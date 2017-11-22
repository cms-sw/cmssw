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
  const std::string dumpFileName = "dump_run000001_event00020137_fed1022.txt";
  std::vector<uint32_t> data;
  onlineMetaData::Data_v1 const* onlineMetaData;
  //OnlineMetaDataRecord onlineMetaDataRecord;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestOnlineMetaDataRecord);


void TestOnlineMetaDataRecord::setUp() {

  if ( data.empty() ) {
    char* cmsswBase;
    cmsswBase = getenv("CMSSW_BASE");
    std::ostringstream dumpFileStr;
    dumpFileStr << cmsswBase << "/src/DataFormats/OnlineMetaData/test/" << dumpFileName;

    std::ifstream dumpFile(dumpFileStr.str().c_str());
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

    CPPUNIT_ASSERT( data.size() == 36 );

    onlineMetaData = reinterpret_cast<onlineMetaData::Data_v1 const*>(data.data());
    //    onlineMetaDataRecord = OnlineMetaDataRecord((unsigned char*)data.data());
    //    std::cout << onlineMetaDataRecord << std::endl;
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
  const uint64_t ts = dcs.getTimestamp().unixTime() * 1000UL + dcs.getTimestamp().microsecondOffset()/1000;
  CPPUNIT_ASSERT_EQUAL( static_cast<uint64_t>(0x15fde021420),ts );
  CPPUNIT_ASSERT_EQUAL( static_cast<uint32_t>(0x019e7dbf),dcs.getHighVoltageReady() );
  CPPUNIT_ASSERT( dcs.highVoltageReady(DCSRecord::Partition::HBHEa) );
  CPPUNIT_ASSERT( ! dcs.highVoltageReady(DCSRecord::Partition::CASTOR) );
  CPPUNIT_ASSERT( dcs.highVoltageReady(DCSRecord::Partition::ESm) );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x468de7eb),dcs.getMagnetCurrent() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x407340c6),dcs.getMagneticField() );
}


void TestOnlineMetaDataRecord::testOnlineBeamSpotRecord() {

  OnlineBeamSpotRecord beamSpot(onlineMetaData->beamSpot);
  std::cout << beamSpot << std::endl;

  // DIP timestamp is in milliseconds
  const uint64_t ts = beamSpot.getTimestamp().unixTime() * 1000UL + beamSpot.getTimestamp().microsecondOffset()/1000;
  CPPUNIT_ASSERT_EQUAL( static_cast<uint64_t>(0x15fdd282e98),ts );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x3db238e8),beamSpot.getX() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0xbd02451f),beamSpot.getY() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x3fb7c654),beamSpot.getZ() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x38f133ac),beamSpot.getDxdz() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x38be5aea),beamSpot.getDydz() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x3e99ddda),beamSpot.getErrX() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x38fb8b6c),beamSpot.getErrY() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x00000000),beamSpot.getErrZ() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x37a38736),beamSpot.getErrDxdz() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x379de6fb),beamSpot.getErrDydz() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x3b9813c8),beamSpot.getWidthX() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x3b8f8833),beamSpot.getWidthY() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x40c36fe7),beamSpot.getSigmaZ() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x394a1529),beamSpot.getErrWidthX() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x394a1529),beamSpot.getErrWidthY() );
  CPPUNIT_ASSERT_EQUAL( castToFloat(0x3e599913),beamSpot.getErrSigmaZ() );
}


void TestOnlineMetaDataRecord::testOnlineLuminosityRecord() {

  OnlineLuminosityRecord lumi(onlineMetaData->luminosity);
  std::cout << lumi << std::endl;

  // DIP timestamp is in milliseconds
  const uint64_t ts = lumi.getTimestamp().unixTime() * 1000UL + lumi.getTimestamp().microsecondOffset()/1000;
  CPPUNIT_ASSERT_EQUAL( static_cast<uint64_t>(0x15fde07acb0),ts );
  CPPUNIT_ASSERT_EQUAL( static_cast<uint16_t>(60),lumi.getLumiSection() );
  CPPUNIT_ASSERT_EQUAL( static_cast<uint16_t>(40),lumi.getLumiNibble() );
  CPPUNIT_ASSERT_EQUAL( static_cast<float>(0),lumi.getInstLumi() );
  CPPUNIT_ASSERT_EQUAL( static_cast<float>(0),lumi.getAvgPileUp() );
}
