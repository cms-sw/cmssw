#ifdef __APPLE__
#include <sys/param.h>
#include <sys/mount.h>
#else
#include <sys/statfs.h>
#endif

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <limits.h>
#include <stdlib.h>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/ResourceMonitorCollection.h"
#include "EventFilter/StorageManager/test/MockAlarmHandler.h"
#include "EventFilter/StorageManager/test/TestHelper.h"

using namespace stor;

class stor::testResourceMonitorCollection : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testResourceMonitorCollection);

  CPPUNIT_TEST(diskSize);
  CPPUNIT_TEST(unknownDisk);
  CPPUNIT_TEST(notMountedDiskAlarm);
  CPPUNIT_TEST(notMountedDiskSuppressAlarm);
  CPPUNIT_TEST(diskUsage);
  CPPUNIT_TEST(processCount);
  CPPUNIT_TEST(processCountWithArguments);

  CPPUNIT_TEST(noSataBeasts);
  CPPUNIT_TEST(sataBeastOkay);
  CPPUNIT_TEST(sataBeastFailed);
  //CPPUNIT_TEST(sataBeastsOnSpecialNode);  // can only be used on specially prepared SATA beast host

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();

  bool notMountedDisk(bool sendAlarm);

  void diskSize();
  void unknownDisk();
  void notMountedDiskAlarm();
  void notMountedDiskSuppressAlarm();
  void diskUsage();
  void processCount();
  void processCountWithArguments();

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
    diskUsage( new ResourceMonitorCollection::DiskUsage() );
  diskUsage->pathName = "/tmp";
  CPPUNIT_ASSERT_THROW( _rmc->retrieveDiskSize(diskUsage), stor::exception::DiskSpaceAlarm );
#ifdef __APPLE__
  struct statfs buf;
  CPPUNIT_ASSERT( statfs(diskUsage->pathName.c_str(), &buf) == 0 );
#else
  struct statfs64 buf;
  CPPUNIT_ASSERT( statfs64(diskUsage->pathName.c_str(), &buf) == 0 );
#endif
  CPPUNIT_ASSERT( buf.f_blocks );
  double diskSize = static_cast<double>(buf.f_blocks * buf.f_bsize) / 1024 / 1024 / 1024;

  CPPUNIT_ASSERT( diskUsage->diskSize > 0 );
  CPPUNIT_ASSERT( diskUsage->diskSize == diskSize );
}


void
testResourceMonitorCollection::unknownDisk()
{
  ResourceMonitorCollection::DiskUsagePtr
    diskUsage( new ResourceMonitorCollection::DiskUsage() );
  diskUsage->pathName = "/aNonExistingDisk";

  CPPUNIT_ASSERT( _ah->noAlarmSet() );
  _rmc->retrieveDiskSize(diskUsage);
  CPPUNIT_ASSERT( diskUsage->diskSize == -1 );
}


bool
testResourceMonitorCollection::notMountedDisk(bool sendAlarm)
{
  const std::string dummyDisk = "/aNonExistingDisk";

  AlarmParams alarmParams;
  alarmParams._isProductionSystem = sendAlarm;
  _rmc->configureAlarms(alarmParams);

  DiskWritingParams dwParams;
  dwParams._nLogicalDisk = 0;
  dwParams._filePath = "/tmp";
  dwParams._highWaterMark = 100;
  dwParams._otherDiskPaths.push_back(dummyDisk);
  _rmc->configureDisks(dwParams);

  _ah->printActiveAlarms("SentinelException");

  std::vector<MockAlarmHandler::Alarms> alarms;
  return _ah->getActiveAlarms("SentinelException", alarms);
}


void testResourceMonitorCollection::notMountedDiskAlarm()
{
  bool alarmsAreSet = notMountedDisk(true);
  CPPUNIT_ASSERT( alarmsAreSet );
}


void testResourceMonitorCollection::notMountedDiskSuppressAlarm()
{
  bool alarmsAreSet = notMountedDisk(false);
  CPPUNIT_ASSERT( !alarmsAreSet );
}


void
testResourceMonitorCollection::diskUsage()
{
  DiskWritingParams dwParams;
  dwParams._nLogicalDisk = 0;
  dwParams._filePath = "/tmp";
  dwParams._highWaterMark = 100;
  dwParams._failHighWaterMark = 100;
  _rmc->configureDisks(dwParams);
  CPPUNIT_ASSERT( _rmc->_diskUsageList.size() == 1 );
  CPPUNIT_ASSERT( _rmc->_nLogicalDisks == 1 );
  ResourceMonitorCollection::DiskUsagePtr diskUsagePtr = _rmc->_diskUsageList[0];
  CPPUNIT_ASSERT( diskUsagePtr.get() != 0 );

#ifdef __APPLE__
  struct statfs buf;
  CPPUNIT_ASSERT( statfs(dwParams._filePath.c_str(), &buf) == 0 );
#else
  struct statfs64 buf;
  CPPUNIT_ASSERT( statfs64(dwParams._filePath.c_str(), &buf) == 0 );
#endif
  CPPUNIT_ASSERT( buf.f_blocks );
  double relDiskUsage = (1 - static_cast<double>(buf.f_bavail) / buf.f_blocks) * 100;

  _rmc->calcDiskUsage();
  ResourceMonitorCollection::Stats stats;
  _rmc->getStats(stats);
  CPPUNIT_ASSERT( stats.diskUsageStatsList.size() == 1 );
  ResourceMonitorCollection::DiskUsageStatsPtr diskUsageStatsPtr = stats.diskUsageStatsList[0];
  CPPUNIT_ASSERT( diskUsageStatsPtr.get() != 0 );

  double statRelDiskUsage = diskUsageStatsPtr->relDiskUsage;
  if (relDiskUsage > 0)
    CPPUNIT_ASSERT( (statRelDiskUsage/relDiskUsage) - 1 < 0.05 );
  else
    CPPUNIT_ASSERT( statRelDiskUsage == relDiskUsage );

  CPPUNIT_ASSERT( diskUsageStatsPtr->alarmState == AlarmHandler::OKAY );
  CPPUNIT_ASSERT( _ah->noAlarmSet() );

  _rmc->_dwParams._highWaterMark = relDiskUsage > 10 ? (relDiskUsage-10) : 0;
  _rmc->calcDiskUsage();
  CPPUNIT_ASSERT( diskUsagePtr->alarmState == AlarmHandler::WARNING );
  CPPUNIT_ASSERT(! _ah->noAlarmSet() );

  _rmc->_dwParams._highWaterMark = (relDiskUsage+10);
  _rmc->calcDiskUsage();
  CPPUNIT_ASSERT( diskUsagePtr->alarmState == AlarmHandler::OKAY );
  CPPUNIT_ASSERT( _ah->noAlarmSet() );
}


void
testResourceMonitorCollection::processCount()
{
  const int processes = 2;

  uid_t myUid = getuid();

  for (int i = 0; i < processes; ++i)
    system("${CMSSW_BASE}/src/EventFilter/StorageManager/test/processCountTest.sh 2> /dev/null &");

  int processCount = _rmc->getProcessCount("processCountTest.sh");
  CPPUNIT_ASSERT( processCount == processes);
  
  processCount = _rmc->getProcessCount("processCountTest.sh", myUid);
  CPPUNIT_ASSERT( processCount == processes);

  processCount = _rmc->getProcessCount("processCountTest.sh", myUid+1);
  CPPUNIT_ASSERT( processCount == 0);

  system("killall -u ${USER} -q sleep");
}


void
testResourceMonitorCollection::processCountWithArguments()
{
  const int processes = 3;

  for (int i = 0; i < processes; ++i)
    system("${CMSSW_BASE}/src/EventFilter/StorageManager/test/processCountTest.sh foo 2> /dev/null &");

  int processCountFoo = _rmc->getProcessCount("processCountTest.sh foo");
  int processCountBar = _rmc->getProcessCount("processCountTest.sh bar");
  CPPUNIT_ASSERT( processCountFoo == processes);
  CPPUNIT_ASSERT( processCountBar == 0);

  system("killall -u ${USER} -q sleep");
}


void
testResourceMonitorCollection::noSataBeasts()
{
  AlarmParams alarmParams;
  alarmParams._isProductionSystem = true;
  _rmc->configureAlarms(alarmParams);

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
  std::ostringstream fileName;
  fileName << getenv("CMSSW_BASE") << "/src/EventFilter/StorageManager/test/SATABeast_okay.html";
  CPPUNIT_ASSERT( testhelper::read_file(fileName.str(), content) );
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
  std::ostringstream fileName;
  fileName << getenv("CMSSW_BASE") << "/src/EventFilter/StorageManager/test/SATABeast_failed.html";
  CPPUNIT_ASSERT( testhelper::read_file(fileName.str(), content) );
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
