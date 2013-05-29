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
  CPPUNIT_TEST(slowDisk);
  CPPUNIT_TEST(slowOtherDisk);
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
  void slowDisk();
  void slowOtherDisk();
  void processCount();
  void processCountWithArguments();

  void noSataBeasts();
  void sataBeastOkay();
  void sataBeastFailed();
  void sataBeastsOnSpecialNode();

private:

  boost::shared_ptr<MockAlarmHandler> ah_;
  ResourceMonitorCollection* rmc_;

};

void
testResourceMonitorCollection::setUp()
{
  ah_.reset(new MockAlarmHandler());
  rmc_ = new ResourceMonitorCollection(boost::posix_time::seconds(1), ah_);
}

void
testResourceMonitorCollection::tearDown()
{
  delete rmc_;
}


void
testResourceMonitorCollection::diskSize()
{
  ResourceMonitorCollection::DiskUsagePtr
    diskUsage( new ResourceMonitorCollection::DiskUsage("/tmp") );
  rmc_->retrieveDiskSize(diskUsage);
  ah_->printActiveAlarms("SentinelException");
  std::vector<MockAlarmHandler::Alarms> alarms;
  CPPUNIT_ASSERT( ah_->getActiveAlarms("SentinelException", alarms) );
  CPPUNIT_ASSERT( alarms.size() == 1 );

#ifdef __APPLE__
  struct statfs buf;
  CPPUNIT_ASSERT( statfs(diskUsage->pathName_.c_str(), &buf) == 0 );
#else
  struct statfs64 buf;
  CPPUNIT_ASSERT( statfs64(diskUsage->pathName_.c_str(), &buf) == 0 );
#endif
  CPPUNIT_ASSERT( buf.f_blocks );
  double diskSize = static_cast<double>(buf.f_blocks * buf.f_bsize) / 1024 / 1024 / 1024;

  CPPUNIT_ASSERT( diskUsage->diskSize_ > 0 );
  CPPUNIT_ASSERT( diskUsage->diskSize_ == diskSize );
}


void
testResourceMonitorCollection::unknownDisk()
{
  ResourceMonitorCollection::DiskUsagePtr
    diskUsage( new ResourceMonitorCollection::DiskUsage("/aNonExistingDisk") );

  CPPUNIT_ASSERT( ah_->noAlarmSet() );
  rmc_->retrieveDiskSize(diskUsage);
  CPPUNIT_ASSERT( diskUsage->diskSize_ == -1 );
}


bool
testResourceMonitorCollection::notMountedDisk(bool sendAlarm)
{
  const std::string dummyDisk = "/aNonExistingDisk";

  AlarmParams alarmParams;
  alarmParams.isProductionSystem_ = sendAlarm;
  rmc_->configureAlarms(alarmParams);

  DiskWritingParams dwParams;
  dwParams.nLogicalDisk_ = 0;
  dwParams.filePath_ = "/tmp";
  dwParams.highWaterMark_ = 100;
  dwParams.otherDiskPaths_.push_back(dummyDisk);
  rmc_->configureDisks(dwParams);

  ah_->printActiveAlarms(dummyDisk);
  
  std::vector<MockAlarmHandler::Alarms> alarms;
  return ah_->getActiveAlarms(dummyDisk, alarms);
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
  dwParams.nLogicalDisk_ = 0;
  dwParams.filePath_ = "/tmp";
  dwParams.highWaterMark_ = 100;
  dwParams.failHighWaterMark_ = 100;
  rmc_->configureDisks(dwParams);
  CPPUNIT_ASSERT( rmc_->diskUsageList_.size() == 1 );
  CPPUNIT_ASSERT( rmc_->nLogicalDisks_ == 1 );
  ResourceMonitorCollection::DiskUsagePtr diskUsagePtr = rmc_->diskUsageList_[0];
  CPPUNIT_ASSERT( diskUsagePtr.get() != 0 );

#ifdef __APPLE__
  struct statfs buf;
  CPPUNIT_ASSERT( statfs(dwParams.filePath_.c_str(), &buf) == 0 );
#else
  struct statfs64 buf;
  CPPUNIT_ASSERT( statfs64(dwParams.filePath_.c_str(), &buf) == 0 );
#endif
  CPPUNIT_ASSERT( buf.f_blocks );
  double relDiskUsage = (1 - static_cast<double>(buf.f_bavail) / buf.f_blocks) * 100;

  rmc_->calcDiskUsage();
  ResourceMonitorCollection::Stats stats;
  rmc_->getStats(stats);
  CPPUNIT_ASSERT( stats.diskUsageStatsList.size() == 1 );
  ResourceMonitorCollection::DiskUsageStatsPtr diskUsageStatsPtr = stats.diskUsageStatsList[0];
  CPPUNIT_ASSERT( diskUsageStatsPtr.get() != 0 );

  double statRelDiskUsage = diskUsageStatsPtr->relDiskUsage;
  if (relDiskUsage > 0)
    CPPUNIT_ASSERT( (statRelDiskUsage/relDiskUsage) - 1 < 0.05 );
  else
    CPPUNIT_ASSERT( statRelDiskUsage == relDiskUsage );

  CPPUNIT_ASSERT( diskUsageStatsPtr->alarmState == AlarmHandler::OKAY );
  CPPUNIT_ASSERT( ah_->noAlarmSet() );

  rmc_->dwParams_.highWaterMark_ = relDiskUsage > 10 ? (relDiskUsage-10) : 0;
  rmc_->calcDiskUsage();
  CPPUNIT_ASSERT( diskUsagePtr->alarmState_ == AlarmHandler::WARNING );
  CPPUNIT_ASSERT(! ah_->noAlarmSet() );

  rmc_->dwParams_.highWaterMark_ = (relDiskUsage+10);
  rmc_->calcDiskUsage();
  CPPUNIT_ASSERT( diskUsagePtr->alarmState_ == AlarmHandler::OKAY );
  CPPUNIT_ASSERT( ah_->noAlarmSet() );
}


void
testResourceMonitorCollection::slowDisk()
{
  const std::string dummyDisk = "/aSlowDiskForUnitTests";

  AlarmParams alarmParams;
  alarmParams.isProductionSystem_ = true;
  rmc_->configureAlarms(alarmParams);

  DiskWritingParams dwParams;
  dwParams.nLogicalDisk_ = 0;
  dwParams.filePath_ = dummyDisk;
  dwParams.highWaterMark_ = 100;
  rmc_->configureDisks(dwParams);

  ah_->printActiveAlarms("SentinelException");
  
  std::vector<MockAlarmHandler::Alarms> alarms;
  CPPUNIT_ASSERT( ah_->getActiveAlarms("SentinelException", alarms) );
  CPPUNIT_ASSERT( alarms.size() == 1 );
}


void
testResourceMonitorCollection::slowOtherDisk()
{
  const std::string dummyDisk = "/aSlowDiskForUnitTests";

  AlarmParams alarmParams;
  alarmParams.isProductionSystem_ = true;
  rmc_->configureAlarms(alarmParams);

  DiskWritingParams dwParams;
  dwParams.nLogicalDisk_ = 0;
  dwParams.filePath_ = "/tmp";
  dwParams.highWaterMark_ = 100;
  dwParams.otherDiskPaths_.push_back(dummyDisk);
  rmc_->configureDisks(dwParams);

  ah_->printActiveAlarms(dummyDisk);
  
  std::vector<MockAlarmHandler::Alarms> alarms;
  CPPUNIT_ASSERT( ah_->getActiveAlarms(dummyDisk, alarms) );
  CPPUNIT_ASSERT( alarms.size() == 1 );
}


void
testResourceMonitorCollection::processCount()
{
  const int processes = 2;

  uid_t myUid = getuid();

  for (int i = 0; i < processes; ++i)
    system("${CMSSW_BASE}/src/EventFilter/StorageManager/test/processCountTest.sh 2> /dev/null &");

  int processCount = rmc_->getProcessCount("processCountTest.sh");
  CPPUNIT_ASSERT( processCount == processes);
  
  processCount = rmc_->getProcessCount("processCountTest.sh", myUid);
  CPPUNIT_ASSERT( processCount == processes);

  processCount = rmc_->getProcessCount("processCountTest.sh", myUid+1);
  CPPUNIT_ASSERT( processCount == 0);

  system("killall -u ${USER} -q sleep");
}


void
testResourceMonitorCollection::processCountWithArguments()
{
  const int processes = 3;

  for (int i = 0; i < processes; ++i)
    system("${CMSSW_BASE}/src/EventFilter/StorageManager/test/processCountTest.sh foo 2> /dev/null &");

  int processCountFoo = rmc_->getProcessCount("processCountTest.sh foo");
  int processCountBar = rmc_->getProcessCount("processCountTest.sh bar");
  CPPUNIT_ASSERT( processCountFoo == processes);
  CPPUNIT_ASSERT( processCountBar == 0);

  system("killall -u ${USER} -q sleep");
}


void
testResourceMonitorCollection::noSataBeasts()
{
  AlarmParams alarmParams;
  alarmParams.isProductionSystem_ = true;
  rmc_->configureAlarms(alarmParams);

  ResourceMonitorCollection::SATABeasts sataBeasts;
  bool foundSataBeasts =
    rmc_->getSataBeasts(sataBeasts);
  CPPUNIT_ASSERT(! foundSataBeasts );
  CPPUNIT_ASSERT( sataBeasts.empty() );

  rmc_->checkSataBeasts();
  CPPUNIT_ASSERT( rmc_->latchedSataBeastStatus_ == -1 );
  CPPUNIT_ASSERT( ah_->noAlarmSet() );

  ResourceMonitorCollection::Stats stats;
  rmc_->getStats(stats);
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
  rmc_->updateSataBeastStatus(sataBeast, content);

  CPPUNIT_ASSERT( rmc_->latchedSataBeastStatus_ == 0 );
  CPPUNIT_ASSERT( ah_->noAlarmSet() );

  ResourceMonitorCollection::Stats stats;
  rmc_->getStats(stats);
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
  rmc_->updateSataBeastStatus(sataBeast, content);
  CPPUNIT_ASSERT(! content.empty() );

  CPPUNIT_ASSERT( rmc_->latchedSataBeastStatus_ == 101 );
  std::vector<MockAlarmHandler::Alarms> alarms;
  bool alarmsAreSet = ah_->getActiveAlarms(sataBeast, alarms);
  CPPUNIT_ASSERT( alarmsAreSet );
  CPPUNIT_ASSERT( alarms.size() == 2 );

  ResourceMonitorCollection::Stats stats;
  rmc_->getStats(stats);
  CPPUNIT_ASSERT( stats.sataBeastStatus == 101 );

  // verify that we can reset the alarms if all is okay
  sataBeastOkay();
}


void
testResourceMonitorCollection::sataBeastsOnSpecialNode()
{
  ResourceMonitorCollection::SATABeasts sataBeasts;
  bool foundSataBeasts =
    rmc_->getSataBeasts(sataBeasts);
  CPPUNIT_ASSERT( foundSataBeasts );
  CPPUNIT_ASSERT( sataBeasts.size() == 1 );
  std::string sataBeast = *(sataBeasts.begin());
  CPPUNIT_ASSERT( sataBeast == "satab-c2c07-06" );

  CPPUNIT_ASSERT(! rmc_->checkSataDisks(sataBeast,"-00.cms") );
  CPPUNIT_ASSERT( rmc_->checkSataDisks(sataBeast,"-10.cms") );

  CPPUNIT_ASSERT( rmc_->latchedSataBeastStatus_ == 101 );

  ResourceMonitorCollection::Stats stats;
  rmc_->getStats(stats);
  CPPUNIT_ASSERT( stats.sataBeastStatus == 101 );

  std::vector<MockAlarmHandler::Alarms> alarms;
  bool alarmsAreSet = ah_->getActiveAlarms(sataBeast, alarms);
  CPPUNIT_ASSERT( alarmsAreSet );

  ah_->printActiveAlarms(sataBeast);
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testResourceMonitorCollection);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
