#include <iostream>
#include <stdlib.h>
#include <string>
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
// This test exercises the DiskWriter thread
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
  CPPUNIT_TEST(writeManyStreams);
  CPPUNIT_TEST(writeSameEventToAllStreams);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();
  void writeAnEvent();
  void writeManyEvents();
  void writeManyStreams();
  void writeSameEventToAllStreams();
  
private:
  void createStreams();
  void destroyStreams();
  void createInitMessage(const std::string& stream);
  stor::I2OChain getAnEvent(const std::string& stream);
  void checkLogFile();
  void checkBoR(const std::string&);
  bool checkFileEntry(const std::string&, unsigned int& fileCount);
  uint32_t checkEoLS(const std::string&);
  void checkEoR(const std::string&);

  static xdaq::Application* app_;
  static boost::shared_ptr<SharedResources> sharedResources_;
  boost::scoped_ptr<DiskWriter> diskWriter_;
  boost::shared_ptr<MockAlarmHandler> alarmHandler_;
  typedef std::vector<std::string> Streams;
  Streams streams_;
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

  streams_.clear();
  streams_.push_back("A");
  streams_.push_back("Calibration");
  streams_.push_back("Express");
  streams_.push_back("HLTMON");
  streams_.push_back("RPCMON");
  streams_.push_back("DQM");

  srand(time(NULL));
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
  
  utils::TimePoint_t startTime = utils::getCurrentTime();
  const unsigned int totalEvents = 12000000;
  for (unsigned int i = 0; i < totalEvents; ++i)
  {
    stor::I2OChain eventMsg = getAnEvent(stream);
    eventMsg.tagForStream(0);
    sharedResources_->streamQueue_->enqWait(eventMsg);
  }
  utils::Duration_t deltaT = utils::getCurrentTime() - startTime;
  std::cout << std::endl << "Wrote " << totalEvents << " events at " << 
    totalEvents/utils::durationToSeconds(deltaT) << " Hz." << std::endl;

  destroyStreams();
  checkLogFile();
}

void testDiskWriter::writeManyStreams()
{
  createStreams();
  for (Streams::const_iterator it = streams_.begin(), itEnd = streams_.end();
       it != itEnd; ++it)
  {
    createInitMessage(*it);
  }
  utils::TimePoint_t startTime = utils::getCurrentTime();
  const unsigned int totalEvents = 12000000;
  const size_t streamCount = streams_.size();
  for (unsigned int i = 0; i < totalEvents; ++i)
  {
    const unsigned int stream = rand() % streamCount;
    stor::I2OChain eventMsg = getAnEvent(streams_[stream]);
    eventMsg.tagForStream(stream);
    sharedResources_->streamQueue_->enqWait(eventMsg);
  }
  utils::Duration_t deltaT = utils::getCurrentTime() - startTime;
  std::cout << std::endl << "Wrote " << totalEvents << " events at " << 
    totalEvents/utils::durationToSeconds(deltaT) << " Hz into " <<
    streamCount << " streams." << std::endl;

  destroyStreams();
  checkLogFile();
}

void testDiskWriter::writeSameEventToAllStreams()
{
  createStreams();
  for (Streams::const_iterator it = streams_.begin(), itEnd = streams_.end();
       it != itEnd; ++it)
  {
    createInitMessage(*it);
  }
  utils::TimePoint_t startTime = utils::getCurrentTime();
  const unsigned int totalEvents = 12000000;
  const size_t streamCount = streams_.size();
  for (unsigned int i = 0; i < totalEvents; ++i)
  {
    stor::I2OChain eventMsg = getAnEvent("A");
    for (size_t i = 0 ; i < streamCount; ++i)
      eventMsg.tagForStream(i);
    sharedResources_->streamQueue_->enqWait(eventMsg);
  }
  utils::Duration_t deltaT = utils::getCurrentTime() - startTime;
  std::cout << std::endl << "Wrote " << totalEvents << " events at " << 
    totalEvents/utils::durationToSeconds(deltaT) << " Hz into all " <<
    streamCount << " streams." << std::endl;

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
  for (Streams::const_iterator it = streams_.begin(), itEnd = streams_.end();
       it != itEnd; ++it)
  {
    EventStreamConfigurationInfo stream(*it,1024,"*",Strings(),*it,1);
    evtCfgList->push_back(stream);
  }
  
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
  while ( diskWriterResources->streamChangeOngoing() && count < 50 )
  {
    utils::sleep(workerParams.DWdeqWaitTime_);
    ++count;
  }
  CPPUNIT_ASSERT( count < 50 );
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
  unsigned int fileCount = 0;

  getline(logfile,line);
  checkBoR(line);
  getline(logfile,line);
  while( checkFileEntry(line,fileCount) )
  {
    getline(logfile,line);
  };
  CPPUNIT_ASSERT( checkEoLS(line) == fileCount );
  getline(logfile,line);
  checkEoR(line);

  logfile.close();
}

void testDiskWriter::checkBoR(const std::string& line)
{
  const boost::regex pattern("^Timestamp:.*run:100.*BoR$");
  std::ostringstream msg;
  msg << "BoR line is wrong: " << line;
  CPPUNIT_ASSERT_MESSAGE( msg.str(), boost::regex_search(line,pattern) );
}

bool testDiskWriter::checkFileEntry(const std::string& line, unsigned int& fileCount)
{
  boost::cmatch match;
  const boost::regex insertPattern("^./insertFile.pl  --FILENAME [\\w\\.]+ --FILECOUNTER .* --CHECKSUM 0 --CHECKSUMIND 0$");
  const boost::regex closePattern("^./closeFile.pl.* --FILENAME ([\\w\\.]+) --FILECOUNTER (\\d+) --NEVENTS (\\d+) --FILESIZE (\\d+) .* --DEBUGCLOSE (\\d+) --CHECKSUM ([0-9a-f]+) --CHECKSUMIND 0$");
      
  if ( boost::regex_search(line,insertPattern) )
  {
    ++fileCount;
    return true;
  }
  if ( boost::regex_match(line.c_str(),match,closePattern) )
  {
    const std::string fileName = match[1];
    const uint32_t fileCounter = atoi(match[2].str().c_str());
    const uint32_t eventCount = atoi(match[3].str().c_str());
    const uint32_t fileSize = atoi(match[4].str().c_str());
    const uint32_t debugClose = atoi(match[5].str().c_str());
    const uint32_t adler32 = strtoul(match[6].str().c_str(), NULL, 16);
    
    const FilesMonitorCollection& fmc =
      sharedResources_->statisticsReporter_->getFilesMonitorCollection();
    FilesMonitorCollection::FileRecordList records;
    fmc.getFileRecords(records);
    FilesMonitorCollection::FileRecordList::const_iterator pos = records.begin();
    while ( pos != records.end() && (*pos)->fileName() != fileName ) ++pos;
    CPPUNIT_ASSERT( pos != records.end() );
    CPPUNIT_ASSERT( fileCounter == (*pos)->fileCounter );
    CPPUNIT_ASSERT( fileSize == (*pos)->fileSize );
    CPPUNIT_ASSERT( eventCount == (*pos)->eventCount );
    CPPUNIT_ASSERT( debugClose == (*pos)->whyClosed );
    CPPUNIT_ASSERT( adler32 == (*pos)->adler32 );

    return true;
  }
  return false;
}

uint32_t testDiskWriter::checkEoLS(const std::string& line)
{
  std::ostringstream streamPattern;
  streamPattern << "^Timestamp:.*run:100.*LS:1\\s+";
  for (Streams::const_iterator it = streams_.begin(), itEnd = streams_.end();
       it != itEnd; ++it)
  {
    streamPattern << *it << ":(\\d)\\s+";
  }
  streamPattern << "EoLS:0$";
  const boost::regex pattern(streamPattern.str());
  boost::cmatch match;
  std::ostringstream msg;
  msg << "EoLS line is wrong: " << line;
  CPPUNIT_ASSERT_MESSAGE( msg.str(), boost::regex_match(line.c_str(),match,pattern) );
  uint32_t fileCount = 0;
  for (unsigned int i = 1; i < streams_.size()+1; ++i)
    fileCount += atoi(match[i].str().c_str());
  return fileCount;  
}

void testDiskWriter::checkEoR(const std::string& line)
{
  const boost::regex pattern("^Timestamp:.*run:100.*LScount:1\\s+EoLScount:0\\s+LastLumi:1\\s+EoR$");
  std::ostringstream msg;
  msg << "EoR line is wrong: " << line;
  CPPUNIT_ASSERT_MESSAGE( msg.str(), boost::regex_search(line,pattern) );
}


// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testDiskWriter);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
