#include <sys/statfs.h>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/ResourceMonitorCollection.h"
#include "EventFilter/StorageManager/test/MockAlarmHandler.h"
#include "EventFilter/StorageManager/test/TestHelper.h"

using namespace stor;

class stor::testResourceMonitorCollection : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testResourceMonitorCollection);

  CPPUNIT_TEST(diskSize);
  CPPUNIT_TEST(unknownDisk);
  CPPUNIT_TEST(notMountedDisk);
  CPPUNIT_TEST(diskUsage);
  CPPUNIT_TEST(processCount);

  CPPUNIT_TEST(noSataBeasts);
  CPPUNIT_TEST(sataBeastOkay);
  CPPUNIT_TEST(sataBeastFailed);
  //CPPUNIT_TEST(sataBeastsOnSpecialNode);  // can only be used on specially prepared SATA beast host

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();

  void diskSize();
  void unknownDisk();
  void notMountedDisk();
  void diskUsage();
  void processCount();

  void noSataBeasts();
  void sataBeastOkay();
  void sataBeastFailed();
  void sataBeastsOnSpecialNode();

private:

  boost::shared_ptr<MockAlarmHandler> _ah;
  ResourceMonitorCollection* _rmc;

};

void
testResourceMonitorCollection::setUp()
{
  _ah.reset(new MockAlarmHandler());
  _rmc = new ResourceMonitorCollection(1, _ah);
}

void
testResourceMonitorCollection::tearDown()
{
  delete _rmc;
}


void
testResourceMonitorCollection::diskSize()
{
  ResourceMonitorCollection::DiskUsagePtr
    diskUsage( new ResourceMonitorCollection::DiskUsage(1) );
  diskUsage->pathName = ".";
  CPPUNIT_ASSERT( diskUsage->retrieveDiskSize() );

  struct statfs64 buf;
  CPPUNIT_ASSERT( statfs64(diskUsage->pathName.c_str(), &buf) == 0 );
  CPPUNIT_ASSERT( buf.f_blocks );
  size_t diskSize = buf.f_blocks * buf.f_bsize / 1024 / 1024 / 1024;

  CPPUNIT_ASSERT( diskUsage->diskSize > 0 );
  CPPUNIT_ASSERT( diskUsage->diskSize == diskSize );
}


void
testResourceMonitorCollection::unknownDisk()
{
  ResourceMonitorCollection::DiskUsagePtr
    diskUsage( new ResourceMonitorCollection::DiskUsage(1) );
  diskUsage->pathName = "/aNonExistingDisk";

  CPPUNIT_ASSERT( _ah->noAlarmSet() );
  CPPUNIT_ASSERT(! diskUsage->retrieveDiskSize() );
  CPPUNIT_ASSERT( diskUsage->diskSize == 0 );
}


void
testResourceMonitorCollection::notMountedDisk()
{
  const std::string dummyDisk = "/aNonExistingDisk";

  DiskWritingParams dwParams;
  dwParams._nLogicalDisk = 0;
  dwParams._filePath = ".";
  dwParams._ecalCalibPath = dummyDisk;
  _rmc->configureDisks(dwParams);

  _ah->printActiveAlarms(dummyDisk);

  std::vector<MockAlarmHandler::Alarms> alarms;
  bool alarmsAreSet = _ah->getActiveAlarms(dummyDisk, alarms);
  CPPUNIT_ASSERT( alarmsAreSet );
}


void
testResourceMonitorCollection::diskUsage()
{
  DiskWritingParams dwParams;
  dwParams._nLogicalDisk = 0;
  dwParams._filePath = ".";
  dwParams._highWaterMark = 1;
  _rmc->configureDisks(dwParams);
  CPPUNIT_ASSERT( _rmc->_diskUsageList.size() == 1 );
  CPPUNIT_ASSERT( _rmc->_latchedNumberOfDisks == 1 );
  ResourceMonitorCollection::DiskUsagePtr diskUsagePtr = _rmc->_diskUsageList[0];
  CPPUNIT_ASSERT( diskUsagePtr.get() != 0 );

  struct statfs64 buf;
  CPPUNIT_ASSERT( statfs64(dwParams._filePath.c_str(), &buf) == 0 );
  CPPUNIT_ASSERT( buf.f_blocks );
  double relDiskUsage = (1 - static_cast<double>(buf.f_bavail) / buf.f_blocks) * 100;

  _rmc->calcDiskUsage();
  ResourceMonitorCollection::Stats stats;
  _rmc->getStats(stats);
  CPPUNIT_ASSERT( stats.diskUsageStatsList.size() == 1 );
  ResourceMonitorCollection::DiskUsageStatsPtr diskUsageStatsPtr = stats.diskUsageStatsList[0];
  CPPUNIT_ASSERT( diskUsageStatsPtr.get() != 0 );

  double statRelDiskUsage = diskUsageStatsPtr->relDiskUsageStats.getLastSampleValue();
  CPPUNIT_ASSERT( (statRelDiskUsage/relDiskUsage) - 1 < 0.01 );

  CPPUNIT_ASSERT( diskUsageStatsPtr->alarmState == AlarmHandler::OKAY );
  CPPUNIT_ASSERT( _ah->noAlarmSet() );

  _rmc->_highWaterMark = (relDiskUsage-10)/100;
  _rmc->calcDiskUsage();
  CPPUNIT_ASSERT( diskUsagePtr->alarmState == AlarmHandler::WARNING );
  CPPUNIT_ASSERT(! _ah->noAlarmSet() );

  _rmc->_highWaterMark = (relDiskUsage+10)/100;
  _rmc->calcDiskUsage();
  CPPUNIT_ASSERT( diskUsagePtr->alarmState == AlarmHandler::OKAY );
  CPPUNIT_ASSERT( _ah->noAlarmSet() );
}


void
testResourceMonitorCollection::processCount()
{
  const int processes = 2;

  for (int i = 0; i < processes; ++i)
    system("sh ./processCountTest.sh &");

  int processCount = _rmc->getProcessCount("processCountTest.sh");

  CPPUNIT_ASSERT( processCount == processes);
}


void
testResourceMonitorCollection::noSataBeasts()
{
  ResourceMonitorCollection::SATABeasts sataBeasts;
  bool foundSataBeasts =
    _rmc->getSataBeasts(sataBeasts);
  CPPUNIT_ASSERT(! foundSataBeasts );
  CPPUNIT_ASSERT( sataBeasts.empty() );

  _rmc->checkSataBeasts();
  CPPUNIT_ASSERT( _rmc->_latchedSataBeastStatus == -1 );
  CPPUNIT_ASSERT( _ah->noAlarmSet() );

  ResourceMonitorCollection::Stats stats;
  _rmc->getStats(stats);
  CPPUNIT_ASSERT( stats.sataBeastStatus == -1 );
}


void
testResourceMonitorCollection::sataBeastOkay()
{
  std::string content;
  std::string sataBeast("test");
  CPPUNIT_ASSERT( testhelper::read_file("SATABeast_okay.html", content) );
  CPPUNIT_ASSERT(! content.empty() );
  _rmc->updateSataBeastStatus(sataBeast, content);

  CPPUNIT_ASSERT( _rmc->_latchedSataBeastStatus == 0 );
  CPPUNIT_ASSERT( _ah->noAlarmSet() );

  ResourceMonitorCollection::Stats stats;
  _rmc->getStats(stats);
  CPPUNIT_ASSERT( stats.sataBeastStatus == 0 );
}


void
testResourceMonitorCollection::sataBeastFailed()
{
  std::string content;
  std::string sataBeast("test");
  CPPUNIT_ASSERT( testhelper::read_file("SATABeast_failed.html", content) );
  _rmc->updateSataBeastStatus(sataBeast, content);
  CPPUNIT_ASSERT(! content.empty() );

  CPPUNIT_ASSERT( _rmc->_latchedSataBeastStatus == 101 );
  std::vector<MockAlarmHandler::Alarms> alarms;
  bool alarmsAreSet = _ah->getActiveAlarms(sataBeast, alarms);
  CPPUNIT_ASSERT( alarmsAreSet );
  CPPUNIT_ASSERT( alarms.size() == 2 );

  ResourceMonitorCollection::Stats stats;
  _rmc->getStats(stats);
  CPPUNIT_ASSERT( stats.sataBeastStatus == 101 );

  // verify that we can reset the alarms if all is okay
  sataBeastOkay();
}


void
testResourceMonitorCollection::sataBeastsOnSpecialNode()
{
  ResourceMonitorCollection::SATABeasts sataBeasts;
  bool foundSataBeasts =
    _rmc->getSataBeasts(sataBeasts);
  CPPUNIT_ASSERT( foundSataBeasts );
  CPPUNIT_ASSERT( sataBeasts.size() == 1 );
  std::string sataBeast = *(sataBeasts.begin());
  CPPUNIT_ASSERT( sataBeast == "satab-c2c07-06" );

  CPPUNIT_ASSERT(! _rmc->checkSataDisks(sataBeast,"-00.cms") );
  CPPUNIT_ASSERT( _rmc->checkSataDisks(sataBeast,"-10.cms") );

  CPPUNIT_ASSERT( _rmc->_latchedSataBeastStatus == 101 );

  ResourceMonitorCollection::Stats stats;
  _rmc->getStats(stats);
  CPPUNIT_ASSERT( stats.sataBeastStatus == 101 );

  std::vector<MockAlarmHandler::Alarms> alarms;
  bool alarmsAreSet = _ah->getActiveAlarms(sataBeast, alarms);
  CPPUNIT_ASSERT( alarmsAreSet );

  _ah->printActiveAlarms(sataBeast);
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testResourceMonitorCollection);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
