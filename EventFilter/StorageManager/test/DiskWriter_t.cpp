#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/regex.hpp"
#include "boost/scoped_ptr.hpp"

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "xcept/tools.h"

#include "DataFormats/Common/interface/HLTenums.h"

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/DiskWriter.h"
#include "EventFilter/StorageManager/interface/DiskWriterResources.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/Utils.h"

#include "EventFilter/StorageManager/test/TestHelper.h"
#include "EventFilter/StorageManager/test/MockAlarmHandler.h"
#include "EventFilter/StorageManager/test/MockApplication.h"


/////////////////////////////////////////////////////////////
//
// This test exercises the DiskWriter class
//
/////////////////////////////////////////////////////////////

using namespace stor;

using stor::testhelper::allocate_frame_with_init_msg;
using stor::testhelper::allocate_frame_with_event_msg;
using stor::testhelper::set_trigger_bit;
using stor::testhelper::clear_trigger_bits;


class testDiskWriter : public CppUnit::TestFixture
{
  typedef toolbox::mem::Reference Reference;
  CPPUNIT_TEST_SUITE(testDiskWriter);
  CPPUNIT_TEST(writeAnEvent);
  CPPUNIT_TEST(writeManyEvents);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();
  void writeAnEvent();
  void writeManyEvents();

private:
  void createStreams();
  void destroyStreams();
  void createInitMessage(const std::string& stream);
  stor::I2OChain getAnEvent(const std::string& stream);
  void checkLogFile();
  bool checkNextFile(std::ifstream&, const unsigned int& fileCount);
  
  static xdaq::Application* app_;
  static boost::shared_ptr<SharedResources> sharedResources_;
  boost::scoped_ptr<DiskWriter> diskWriter_;
  boost::shared_ptr<MockAlarmHandler> alarmHandler_;

  std::string path;
};

xdaq::Application* testDiskWriter::app_;
boost::shared_ptr<SharedResources> testDiskWriter::sharedResources_;

void testDiskWriter::setUp()
{
  alarmHandler_.reset(new MockAlarmHandler());
  if (sharedResources_.get() == 0)
  {
    app_ = mockapps::getMockXdaqApplication();
    sharedResources_.reset( new SharedResources() );
    sharedResources_->configuration_.reset(new Configuration(app_->getApplicationInfoSpace(), 0));
    sharedResources_->statisticsReporter_.reset(new StatisticsReporter(app_,sharedResources_));
  }
  sharedResources_->statisticsReporter_->reset();
  sharedResources_->initMsgCollection_.reset(new InitMsgCollection());
  sharedResources_->alarmHandler_ = alarmHandler_;
  sharedResources_->streamQueue_.reset(new StreamQueue(1024));
  sharedResources_->diskWriterResources_.reset(new DiskWriterResources());
  diskWriter_.reset(new DiskWriter(app_, sharedResources_));
  diskWriter_->startWorkLoop("theDiskWriter");
  
  std::ostringstream spath;
  spath << "/tmp/smtest_" << getpid();
  path = spath.str();
  boost::filesystem::remove_all(path);
  CPPUNIT_ASSERT( boost::filesystem::create_directory(path) );
  CPPUNIT_ASSERT( boost::filesystem::create_directory(path + "/open") );
  CPPUNIT_ASSERT( boost::filesystem::create_directory(path + "/closed") );
}

void testDiskWriter::tearDown()
{
  std::vector<MockAlarmHandler::Alarms> alarms;
  std::ostringstream msg;
  if ( alarmHandler_->getActiveAlarms("SentinelException", alarms) )
  {
    msg << "Active alarms: " << std::endl;
    for (std::vector<MockAlarmHandler::Alarms>::iterator
           it = alarms.begin(), itEnd = alarms.end();
         it != itEnd; ++it)
      msg << xcept::stdformat_exception_history(it->second) << std::endl;
  }
  else
  {
    msg << "Non-sentinel alarm was raised" << std::endl;
  }
  CPPUNIT_ASSERT_MESSAGE( msg.str(), alarmHandler_->noAlarmSet() );
  diskWriter_.reset();
  CPPUNIT_ASSERT( boost::filesystem::remove_all(path) > 0 );
}

void testDiskWriter::writeAnEvent()
{
  const std::string stream = "A";
  createStreams();
  createInitMessage(stream);
  stor::I2OChain eventMsg = getAnEvent(stream);
  eventMsg.tagForStream(0);
  sharedResources_->streamQueue_->enqWait(eventMsg);

  destroyStreams();
  checkLogFile();
}

void testDiskWriter::writeManyEvents()
{
  const std::string stream = "A";
  createStreams();
  createInitMessage(stream);
  for (unsigned int i = 0; i < 12000000; ++i)
  {
    stor::I2OChain eventMsg = getAnEvent(stream);
    eventMsg.tagForStream(0);
    sharedResources_->streamQueue_->enqWait(eventMsg);
  }

  destroyStreams();
  checkLogFile();
}

void testDiskWriter::createStreams()
{
  DiskWritingParams dwParams =
    sharedResources_->configuration_->getDiskWritingParams();
  WorkerThreadParams workerParams =
    sharedResources_->configuration_->getWorkerThreadParams();
  
  EvtStrConfigListPtr evtCfgList(new EvtStrConfigList);
  ErrStrConfigListPtr errCfgList(new ErrStrConfigList);
  EventStreamConfigurationInfo streamA("A",1024,"*",Strings(),"A",1);
  evtCfgList->push_back(streamA);
  
  sharedResources_->configuration_->setCurrentEventStreamConfig(evtCfgList);
  sharedResources_->configuration_->setCurrentErrorStreamConfig(errCfgList);

  dwParams.filePath_ = path;
  dwParams.dbFilePath_ = path;
  dwParams.checkAdler32_ = true;
  
  DiskWriterResourcesPtr diskWriterResources =
    sharedResources_->diskWriterResources_;
  diskWriterResources->requestStreamConfiguration(
    evtCfgList, errCfgList,
    dwParams, 100,
    workerParams.DWdeqWaitTime_);
  
  unsigned int count = 0;
  while ( diskWriterResources->streamChangeOngoing() && count < 5 )
  {
    utils::sleep(workerParams.DWdeqWaitTime_);
    ++count;
  }
  CPPUNIT_ASSERT( count < 5 );
}

void testDiskWriter::destroyStreams()
{
  WorkerThreadParams workerParams =
    sharedResources_->configuration_->getWorkerThreadParams();
  utils::sleep(workerParams.DWdeqWaitTime_);
  CPPUNIT_ASSERT( sharedResources_->streamQueue_->empty() );
  
  DiskWriterResourcesPtr diskWriterResources =
    sharedResources_->diskWriterResources_;
  diskWriterResources->requestStreamDestruction();

  unsigned int count = 0;
  while ( diskWriterResources->streamChangeOngoing() && count < 5 )
  {
    utils::sleep(workerParams.DWdeqWaitTime_);
    ++count;
  }
  CPPUNIT_ASSERT( count < 5 );
}

void testDiskWriter::createInitMessage(const std::string& stream)
{
  Reference* ref = allocate_frame_with_init_msg(stream);
  stor::I2OChain initMsg(ref);
  std::vector<unsigned char> b;
  initMsg.copyFragmentsIntoBuffer(b);
  InitMsgView imv( &b[0] );
  CPPUNIT_ASSERT( sharedResources_->initMsgCollection_->addIfUnique(imv) );
  CPPUNIT_ASSERT( sharedResources_->initMsgCollection_->getElementForOutputModule(stream).get() != 0 );
}

stor::I2OChain testDiskWriter::getAnEvent(const std::string& stream)
{
  uint32_t eventNumber = 1;
  uint32_t hltBitCount = 9;
  std::vector<unsigned char> hltBits;
  clear_trigger_bits(hltBits);
  set_trigger_bit(hltBits, 0, edm::hlt::Pass);

  Reference* ref = allocate_frame_with_event_msg("A", hltBits, hltBitCount, eventNumber);
  return stor::I2OChain(ref);
}

void testDiskWriter::checkLogFile()
{
  DiskWritingParams dwParams =
    sharedResources_->configuration_->getDiskWritingParams();

  std::ostringstream dbfilename;
  dbfilename
    << path // Don't use dwParams.dbFilePath_ here. The configuration does not reflect the actual path.
      << "/"
      << utils::dateStamp(utils::getCurrentTime())
      << "-" << dwParams.hostName_
      << "-" << dwParams.smInstanceString_
      << ".log";

  std::ifstream logfile;
  logfile.open(dbfilename.str());
  CPPUNIT_ASSERT( logfile.is_open() );

  std::string line;
  {
    getline(logfile,line);
    const boost::regex pattern("^Timestamp:.*run:100.*BoR$");
    std::ostringstream msg;
    msg << "BoR line of logfile " << dbfilename.str() << " is wrong: " << line;
    CPPUNIT_ASSERT_MESSAGE( msg.str(), boost::regex_search(line,pattern) );
  }
  unsigned int fileCount = 1;
  while ( checkNextFile(logfile,fileCount) ) ++fileCount;
  {
    getline(logfile,line);
    const boost::regex pattern("^Timestamp:.*run:100.*LS:1\\s+A:(\\d)\\s+EoLS:0$");
    boost::cmatch match;
    std::ostringstream msg;
    msg << "EoLS line of logfile " << dbfilename.str() << " is wrong: " << line;
    CPPUNIT_ASSERT_MESSAGE( msg.str(), boost::regex_match(line.c_str(),match,pattern) );
    const uint32_t fileCountInLog = atoi(match[1].str().c_str());
    CPPUNIT_ASSERT( fileCountInLog == fileCount );
  }
  {
    getline(logfile,line);
    const boost::regex pattern("^Timestamp:.*run:100.*LScount:1\\s+EoLScount:0\\s+LastLumi:1\\s+EoR$");
    std::ostringstream msg;
    msg << "BoR line of logfile " << dbfilename.str() << " is wrong: " << line;
    CPPUNIT_ASSERT_MESSAGE( msg.str(), boost::regex_search(line,pattern) );
  }
  logfile.close();
}

bool testDiskWriter::checkNextFile(std::ifstream& logfile, const unsigned int& fileCount)
{
  std::string line;
  boost::cmatch match;
  {
    getline(logfile,line);
    const boost::regex pattern("^./insertFile.pl  --FILENAME ([\\w\\.]+) --FILECOUNTER .* --CHECKSUM 0 --CHECKSUMIND 0$");
    std::ostringstream msg;
    msg << "Insert line is wrong: " << line;
    CPPUNIT_ASSERT_MESSAGE( msg.str(), boost::regex_match(line.c_str(),match,pattern) );
  }
  const std::string filename = match[1];
  {
    getline(logfile,line);
    const boost::regex pattern("^./closeFile.pl.* --FILECOUNTER (\\d+) --NEVENTS (\\d+) --FILESIZE (\\d+) .* --DEBUGCLOSE (\\d+) --CHECKSUM ([0-9a-f]+) --CHECKSUMIND 0$");
    std::ostringstream msg;
    msg << "Close line is wrong: " << line;
    CPPUNIT_ASSERT_MESSAGE( msg.str(), boost::regex_match(line.c_str(),match,pattern) );
  }
  const uint32_t fileCounter = atoi(match[1].str().c_str());
  const uint32_t eventCount = atoi(match[2].str().c_str());
  const uint32_t fileSize = atoi(match[3].str().c_str());
  const uint32_t debugClose = atoi(match[4].str().c_str());
  const uint32_t adler32 = strtoul(match[5].str().c_str(), NULL, 16);
  
  const FilesMonitorCollection& fmc =
    sharedResources_->statisticsReporter_->getFilesMonitorCollection();
  FilesMonitorCollection::FileRecordList records;
  fmc.getFileRecords(records);
  CPPUNIT_ASSERT( filename == records[fileCount-1]->fileName() );
  CPPUNIT_ASSERT( fileCounter == records[fileCount-1]->fileCounter );
  CPPUNIT_ASSERT( fileSize == records[fileCount-1]->fileSize );
  CPPUNIT_ASSERT( eventCount == records[fileCount-1]->eventCount );
  CPPUNIT_ASSERT( adler32 == records[fileCount-1]->adler32 );

  return (debugClose == 4);
}


// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testDiskWriter);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
