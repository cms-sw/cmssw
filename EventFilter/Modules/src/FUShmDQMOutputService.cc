/**
 * This class is responsible for collecting data quality monitoring (DQM)
 * information, packaging it in DQMEvent objects, and writing them out to
 * shared memory for the Resource Broker to send to the Storage Manager
 *
 * 27-Dec-2006 - KAB  - Initial Implementation
 * 29-Mar-2007 - HWKC - changes for shared memory usage
 *
 * Reference code can be found in the following files:
 * - IOPool/Streamer/interface/StreamerOutputModule.h
 * - IOPool/Streamer/interface/StreamerSerializer.h
 * - IOPool/Streamer/src/StreamerSerializer.cc
 * - EventFilter/FUSender/src/FUStreamerI2OWriter.h
 * - EventFilter/FUSender/src/FUStreamerI2OWriter.cc
 * - DQMServices/Daemon/src/MonitorDaemon.cc
 * - FWCore/ServiceRegistry/interface/ActivityRegistry.h
 * - FWCore/ServiceRegistry/src/ActivityRegistry.cc
 * - DQMServices/NodeROOT/src/SenderBase.cc
 * - DQMServices/NodeROOT/src/ReceiverBase.cc
 *
 * $Id: FUShmDQMOutputService.cc,v 1.24 2012/05/02 15:02:19 smorovic Exp $
 */

#include "EventFilter/Modules/interface/FUShmDQMOutputService.h"
#include "EventFilter/Utilities/interface/MicroStateService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Utilities/src/Guid.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "TClass.h"
#include "zlib.h"
#include <unistd.h>
#include <sys/types.h>

using namespace std;

/**
 * Local debug flag.  0 => no debug output; 1 => tracing of necessary
 * routines; 2 => additional tracing of many framework signals;
 * 3 => deserialization test.
 */
#define DSS_DEBUG 0

/**
 * Initialize the static variables for the filter unit indentifiers.
 */
bool FUShmDQMOutputService::fuIdsInitialized_ = false;
uint32 FUShmDQMOutputService::fuGuidValue_ = 0;

/**
 * FUShmDQMOutputService constructor.
 */
FUShmDQMOutputService::FUShmDQMOutputService(const edm::ParameterSet &pset,
                                   edm::ActivityRegistry &actReg)
  : evf::ServiceWeb("FUShmDQMOutputService")
  , updateNumber_(0)
  , shmBuffer_(0)
  , nbUpdates_(0)
  , input("INPUT")
  , dqm("DQM")
  , attach_(false)
{

  // specify the routine to be called after event processing.  This routine
  // will be used to periodically fetch monitor elements from the DQM
  // backend and write out to shared memory for sending to the storage manager.
  actReg.watchPostEndLumi(this, &FUShmDQMOutputService::postEndLumi);

  // specify the routine to be called after the input source has been
  // constructed.  This routine will be used to initialize our connection
  // to the storage manager and any other needed setup.??
  actReg.watchPostSourceConstruction(this,
         &FUShmDQMOutputService::postSourceConstructionProcessing);

  // specify the routine to be called when a run begins
  actReg.watchPreBeginRun(this, &FUShmDQMOutputService::preBeginRun);

  // specify the routine to be called when the job has finished.  It will
  // be used to disconnect from the SM, if needed, and any other shutdown
  // tasks that are needed.??
  actReg.watchPostEndJob(this, &FUShmDQMOutputService::postEndJobProcessing);

  // set internal values from the parameter set
  int initialSize =
    pset.getUntrackedParameter<int>("initialMessageBufferSize", 1000000);
  messageBuffer_.resize(initialSize);
  lumiSectionsPerUpdate_ = pset.getParameter<double>("lumiSectionsPerUpdate");
  // for the moment, only support a number of lumi sections per update >= 1
  if (lumiSectionsPerUpdate_ <= 1.0) {lumiSectionsPerUpdate_ = 1.0;}
  initializationIsNeeded_ = true;
  useCompression_ = pset.getParameter<bool>("useCompression");
  compressionLevel_ = pset.getParameter<int>("compressionLevel");
  // the default for lumiSectionInterval_ is 0, meaning get it from the event
  // otherwise we get a fake one that should match the fake lumi block
  // for events (if any) as long as the time between lumi blocks is larger
  // than the time difference between creating this service and the 
  // FUShmOutputModule event output module
  lumiSectionInterval_ =
    pset.getUntrackedParameter<int>("lumiSectionInterval", 0); // seconds
  if (lumiSectionInterval_ < 1) {lumiSectionInterval_ = 0;}

  // for fake test luminosity sections
  struct timeval now;
  struct timezone dummyTZ;
  gettimeofday(&now, &dummyTZ);
  // we will count lumi section numbers from this time
  timeInSecSinceUTC_ = static_cast<double>(now.tv_sec) + (static_cast<double>(now.tv_usec)/1000000.0);

  int got_host = gethostname(host_name_, sizeof(host_name_));
  if(got_host != 0) strcpy(host_name_, "noHostNameFoundOrTooLong");

  if (! fuIdsInitialized_) {
    fuIdsInitialized_ = true;

    edm::Guid guidObj(true);
    std::string guidString = guidObj.toString();
    //std::cout << "DQMOutput GUID string = " << guidString << std::endl;

    uLong crc = crc32(0L, Z_NULL, 0);
    Bytef* buf = (Bytef*)guidString.data();
    crc = crc32(crc, buf, guidString.length());
    fuGuidValue_ = crc;

    //std::cout << "DQMOutput GUID value = 0x" << std::hex << fuGuidValue_ << std::endl;
  }
}

/**
 * FUShmDQMOutputService destructor.
 */
FUShmDQMOutputService::~FUShmDQMOutputService(void)
{
  shmdt(shmBuffer_);
}

void FUShmDQMOutputService::defaultWebPage(xgi::Input *in, xgi::Output *out)
{
}

void FUShmDQMOutputService::publish(xdata::InfoSpace *is)
{
  try{
    is->fireItemAvailable("nbDqmUpdates",&nbUpdates_);
  }
  catch(xdata::exception::Exception &e){
    edm::LogInfo("FUShmDQMOutputService")
      << " exception when publishing to InfoSpace "; 
  } 
}

void FUShmDQMOutputService::postEndLumi(edm::LuminosityBlock const &lb, edm::EventSetup const &es)
{
  if (attach_) attachToShm();
  attach_=false;
  
  evf::MicroStateService *mss = 0;
  try{
    mss = edm::Service<evf::MicroStateService>().operator->();
    if(mss) mss->setMicroState(&dqm);
  }
  catch(...) { 
    edm::LogError("FUShmDQMOutputService")<< "exception when trying to get service MicroStateService";
  }
  

  // fake the luminosity section if we don't want to use the real one
  unsigned int thisLumiSection = 0;
  if(lumiSectionInterval_ == 0)
    thisLumiSection = lb.luminosityBlock();
  else {
    // match the code in Event output module to get the same (almost) lumi number
    struct timeval now;
    struct timezone dummyTZ;
    gettimeofday(&now, &dummyTZ);
    double timeInSec = static_cast<double>(now.tv_sec) + (static_cast<double>(now.tv_usec)/1000000.0) - timeInSecSinceUTC_;
    // what about overflows?
    if(lumiSectionInterval_ > 0) thisLumiSection = static_cast<uint32>(timeInSec/lumiSectionInterval_);
  }

   // special handling for the first event
  if (initializationIsNeeded_) {
    initializationIsNeeded_ = false;
    lumiSectionOfPreviousUpdate_ = thisLumiSection;
    firstLumiSectionSeen_ = thisLumiSection;

    // for when a run(job) had ended and we start a new run(job)
    // for fake test luminosity sections
    struct timeval now;
    struct timezone dummyTZ;
    gettimeofday(&now, &dummyTZ);
    // we will count lumi section numbers from this time
    timeInSecSinceUTC_ = static_cast<double>(now.tv_sec) + (static_cast<double>(now.tv_usec)/1000000.0);
  }

  //  std::cout << getpid() << ": :" //<< gettid() 
  //	    << ":DQMOutputService check if have to send update for lumiSection " << thisLumiSection << std::endl;
  if(thisLumiSection%4!=0) 
    {
//       std::cout << getpid() << ": :" //<< gettid() 
// 		<< ":DQMOutputService skipping update for lumiSection " << thisLumiSection << std::endl;
      if(mss) mss->setMicroState(&input);
      return;
    }
//   std::cout << getpid() << ": :" //<< gettid() 
// 	    << ":DQMOutputService sending update for lumiSection " << thisLumiSection << std::endl;
  // Calculate the update ID and lumi ID for this update
  // fullUpdateRatio and fullLsDelta are unused. comment out the calculation.
  //int fullLsDelta = (int) (thisLumiSection - firstLumiSectionSeen_);
  //double fullUpdateRatio = ((double) fullLsDelta) / lumiSectionsPerUpdate_;
  // this is the update number starting from zero

  // this is the actual luminosity section number for the beginning lumi section of this update
  unsigned int lumiSectionTag = thisLumiSection;

  // retry the lookup of the backend interface, if needed
  if (bei == NULL) {
    bei = edm::Service<DQMStore>().operator->();
  }

  // to go any further, a backend interface pointer is crucial
  if (bei == NULL) {
    throw cms::Exception("postEventProcessing", "FUShmDQMOutputService")
      << "Unable to lookup the DQMStore service!\n";
  }

  // determine the top level folders (these will be used for grouping
  // monitor elements into DQM Events)
  std::vector<std::string> topLevelFolderList;
  //std::cout << "### SenderService, pwd = " << bei->pwd() << std::endl;
  bei->cd();
  //std::cout << "### SenderService, pwd = " << bei->pwd() << std::endl;
  topLevelFolderList = bei->getSubdirs();

  // find the monitor elements under each top level folder (including
  // subdirectories)
  std::map< std::string, DQMEvent::TObjectTable > toMap;
  std::vector<std::string>::const_iterator dirIter;
  for (dirIter = topLevelFolderList.begin();
       dirIter != topLevelFolderList.end();
       dirIter++) {
    std::string dirName = *dirIter;
    DQMEvent::TObjectTable toTable;

    // find the MEs
    findMonitorElements(toTable, dirName);

    // store the list in the map
    toMap[dirName] = toTable;
  }

  // create a DQMEvent message for each top-level folder
  // and write each to the shared memory
  for (dirIter = topLevelFolderList.begin();
       dirIter != topLevelFolderList.end();
       dirIter++) {
    std::string dirName = *dirIter;
    DQMEvent::TObjectTable toTable = toMap[dirName];
    if (toTable.size() == 0) {continue;}

    // serialize the monitor element data
    serializeWorker_.serializeDQMEvent(toTable, useCompression_,
                                       compressionLevel_);

    // resize the message buffer, if needed 
    unsigned int srcSize = serializeWorker_.currentSpaceUsed();
    unsigned int newSize = srcSize + 50000;  // allow for header
    if (messageBuffer_.size() < newSize) messageBuffer_.resize(newSize);

    // create the message
    DQMEventMsgBuilder dqmMsgBuilder(&messageBuffer_[0], messageBuffer_.size(),
                                     lb.run(), lb.luminosityBlock(),
				     lb.endTime(),
                                     lumiSectionTag, updateNumber_,
                                     (uint32)serializeWorker_.adler32_chksum(),
                                     host_name_,
                                     edm::getReleaseVersion(), dirName,
                                     toTable);

    // copy the serialized data into the message
    unsigned char* src = serializeWorker_.bufferPointer();
    std::copy(src,src + srcSize, dqmMsgBuilder.eventAddress());
    dqmMsgBuilder.setEventLength(srcSize);
    if (useCompression_) {
      dqmMsgBuilder.setCompressionFlag(serializeWorker_.currentEventSize());
    }

    // write the filter unit UUID and PID into the message
    dqmMsgBuilder.setFUProcessId(getpid());
    dqmMsgBuilder.setFUGuid(fuGuidValue_);

    // send the message
    writeShmDQMData(dqmMsgBuilder);
//     std::cout << getpid() << ": :" // << gettid() 
// 	      << ":DQMOutputService DONE sending update for lumiSection " << thisLumiSection << std::endl;
    if(mss) mss->setMicroState(&input);

  }
  
  // reset monitor elements that have requested it
  // TODO - enable this
  //bei->doneSending(true, true);
  
  // update the "previous" lumi section
  lumiSectionOfPreviousUpdate_ = thisLumiSection;
  nbUpdates_++;
  updateNumber_++;
}

/**
 * Callback to be used after the input source has been constructed.  It
 * takes care of any intialization that is needed by this service.
 */
void FUShmDQMOutputService::postSourceConstructionProcessing(const edm::ModuleDescription &moduleDesc)
{

  bei = edm::Service<DQMStore>().operator->();
}

/**
 * Callback to be used after the Run has been created by the InputSource
 * but before any modules have seen the Run
 */
void FUShmDQMOutputService::preBeginRun(const edm::RunID &runID,
                                        const edm::Timestamp &timestamp)
{
  nbUpdates_ = 0;
  updateNumber_ = 0;
  initializationIsNeeded_ = true;
}

/**
 * Callback to be used after the end job operation has finished.  It takes
 * care of any necessary cleanup.
 */
void FUShmDQMOutputService::postEndJobProcessing()
{
  // since the service is not destroyed we need to take care of endjob items here
  initializationIsNeeded_ = true;
}

/**
 * Finds all of the monitor elements under the specified folder,
 * including those in subdirectories.
 */
void FUShmDQMOutputService::findMonitorElements(DQMEvent::TObjectTable &toTable,
                                           std::string folderPath)
{
  if (bei == NULL) {return;}

  // fetch the monitor elements in the specified directory
  std::vector<MonitorElement *> localMEList = bei->getContents(folderPath);
  //MonitorElementRootFolder* folderPtr = bei->getDirectory(folderPath);

  // add the MEs that should be updated to the table
  std::vector<TObject *> updateTOList;
  for (int idx = 0; idx < (int) localMEList.size(); idx++) {
    MonitorElement *mePtr = localMEList[idx];
    //    if (mePtr->wasUpdated()) { // @@EM send updated and not (to be revised)
    updateTOList.push_back(mePtr->getRootObject());
      //    }
  }
  if (updateTOList.size() > 0) {
    toTable[folderPath] = updateTOList;
  }

  // find the subdirectories in this folder
  // (checking if the directory exists is probably overkill,
  // but we really don't want to create new folders using
  // setCurrentFolder())
  if (bei->dirExists(folderPath)) {
    bei->setCurrentFolder(folderPath);
    std::vector<std::string> subDirList = bei->getSubdirs();

    // loop over the subdirectories, find the MEs in each one
    std::vector<std::string>::const_iterator dirIter;
    for (dirIter = subDirList.begin(); dirIter != subDirList.end(); dirIter++) {
      std::string subDirPath = (*dirIter);
      findMonitorElements(toTable, subDirPath);
    }
  }
}

/**
 * Writes the specified DQM event message to shared memory.
 */
void FUShmDQMOutputService::writeShmDQMData(DQMEventMsgBuilder const& dqmMsgBuilder)
{
  // fetch the location and size of the message buffer
  unsigned char* buffer = (unsigned char*) dqmMsgBuilder.startAddress();
  unsigned int size = dqmMsgBuilder.size();

  // fetch the run, event, and folder number for addition to the I2O fragments
  DQMEventMsgView dqmMsgView(buffer);
  unsigned int runid = dqmMsgView.runNumber();
  unsigned int eventid = dqmMsgView.eventNumberAtUpdate();

  // We need to generate an unique 32 bit ID from the top folder name
  std::string topFolder = dqmMsgView.topFolderName();
  uLong crc = crc32(0L, Z_NULL, 0);
  Bytef* buf = (Bytef*)topFolder.data();
  crc = crc32(crc, buf, topFolder.length());

  if(!shmBuffer_) {
    edm::LogError("FUDQMShmOutputService") 
      << " Error writing to shared memory as shm is not available";
  } else {
    bool ret = shmBuffer_->writeDqmEventData(runid, eventid, (unsigned int)crc,
                                             getpid(), fuGuidValue_, buffer, size);
    if(!ret) edm::LogError("FUShmDQMOutputService") << " Error with writing data to ShmBuffer";
  }

}

void FUShmDQMOutputService::setAttachToShm() {
  attach_=true;
}

bool FUShmDQMOutputService::attachToShm()
{
  if(0==shmBuffer_) {
    shmBuffer_ = evf::FUShmBuffer::getShmBuffer();
    if (0==shmBuffer_) {
      edm::LogError("FUDQMShmOutputService")<<"Failed to attach to shared memory";
      return false;
    }
    return true;    
  }
  return false;

}



bool FUShmDQMOutputService::detachFromShm()
{
  if(0!=shmBuffer_) {
    shmdt(shmBuffer_);
    shmBuffer_ = 0;
  }
  return true;
}
