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
 * $Id: FUShmDQMOutputService.cc,v 1.12 2009/05/08 13:46:36 biery Exp $
 */

#include "EventFilter/Modules/interface/FUShmDQMOutputService.h"
#include "EventFilter/Utilities/interface/MicroStateService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Utilities/src/Guid.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "TClass.h"
#include "zlib.h"

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
uint32 FUShmDQMOutputService::fuProcId_ = 0;
uint32 FUShmDQMOutputService::fuGuidValue_ = 0;

/**
 * FUShmDQMOutputService constructor.
 */
FUShmDQMOutputService::FUShmDQMOutputService(const edm::ParameterSet &pset,
                                   edm::ActivityRegistry &actReg):
  shmBuffer_(0)
{
  if (DSS_DEBUG) {cout << "FUShmDQMOutputService Constructor" << endl;}

  // specify the routine to be called after event processing.  This routine
  // will be used to periodically fetch monitor elements from the DQM
  // backend and write out to shared memory for sending to the storage manager.
  actReg.watchPostProcessEvent(this, &FUShmDQMOutputService::postEventProcessing);

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

  // helpful callbacks when trying to understand the signals that are
  // available to framework services
  if (DSS_DEBUG >= 2) {
    actReg.watchPostBeginJob(this, &FUShmDQMOutputService::postBeginJobProcessing);
    actReg.watchPreSource(this, &FUShmDQMOutputService::preSourceProcessing);
    actReg.watchPostSource(this, &FUShmDQMOutputService::postSourceProcessing);
    actReg.watchPreModule(this, &FUShmDQMOutputService::preModuleProcessing);
    actReg.watchPostModule(this, &FUShmDQMOutputService::postModuleProcessing);
    actReg.watchPreSourceConstruction(this,
           &FUShmDQMOutputService::preSourceConstructionProcessing);
    actReg.watchPreModuleConstruction(this,
           &FUShmDQMOutputService::preModuleConstructionProcessing);
    actReg.watchPostModuleConstruction(this,
           &FUShmDQMOutputService::postModuleConstructionProcessing);
  }

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

  if (! fuIdsInitialized_) {
    fuIdsInitialized_ = true;

    edm::Guid guidObj(true);
    std::string guidString = guidObj.toString();
    //std::cout << "DQMOutput GUID string = " << guidString << std::endl;

    uLong crc = crc32(0L, Z_NULL, 0);
    Bytef* buf = (Bytef*)guidString.data();
    crc = crc32(crc, buf, guidString.length());
    fuGuidValue_ = crc;

    fuProcId_ = getpid();
    //std::cout << "DQMOutput GUID value = 0x" << std::hex << fuGuidValue_ << std::dec
    //          << " for PID = " << fuProcId_ << std::endl;
  }
}

/**
 * FUShmDQMOutputService destructor.
 */
FUShmDQMOutputService::~FUShmDQMOutputService(void)
{
  if (DSS_DEBUG) {cout << "FUShmDQMOutputService Destructor" << endl;}
  shmdt(shmBuffer_);
}

/**
 * Callback to be used after event processing has finished.  (The
 * "post event" signal is generated after all of the analysis modules
 * have run <b>and</b> any output modules have run.)  This routine is
 * used to periodically gather monitor elements from the DQM backend
 * and send them to the storage manager.
 */
void FUShmDQMOutputService::postEventProcessing(const edm::Event &event,
                                           const edm::EventSetup &eventSetup)
{
  std::string dqm = "DQM";
  evf::MicroStateService *mss = 0;
  try{
    mss = edm::Service<evf::MicroStateService>().operator->();
  }
  catch(...) { 
    edm::LogError("FUShmDQMOutputService")<< "exception when trying to get service MicroStateService";
  }
  mss->setMicroState(dqm);

  // fake the luminosity section if we don't want to use the real one
  unsigned int thisLumiSection = 0;
  if(lumiSectionInterval_ == 0)
    thisLumiSection = event.luminosityBlock();
  else {
    // match the code in Event output module to get the same (almost) lumi number
    struct timeval now;
    struct timezone dummyTZ;
    gettimeofday(&now, &dummyTZ);
    double timeInSec = static_cast<double>(now.tv_sec) + (static_cast<double>(now.tv_usec)/1000000.0) - timeInSecSinceUTC_;
    // what about overflows?
    if(lumiSectionInterval_ > 0) thisLumiSection = static_cast<uint32>(timeInSec/lumiSectionInterval_);
  }

  if (DSS_DEBUG) {
    cout << "FUShmDQMOutputService::postEventProcessing called, event number "
         << event.id().event() << ", lumi section "
         << thisLumiSection << endl;
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

  // We send a DQMEvent when the correct number of luminosity sections have passed
  // but this will occur here for the first event of a new lumi section which
  // means the data for the first event of this new lumi section is always added to the
  // to the DQM data for the update for the previous lumi section - beware!
  // Can only correct in this postEventProcessing stage if we knew this is the last
  // event of a lumi section. (There is no preEventProcessing possibility?)

  // only continue if the correct number of luminosity sections have passed
  int lsDelta = (int) (thisLumiSection - lumiSectionOfPreviousUpdate_);
  double updateRatio = ((double) lsDelta) / lumiSectionsPerUpdate_;
  if (updateRatio < 1.0) {return;}

  // CAlculate the update ID and lumi ID for this update
  int fullLsDelta = (int) (thisLumiSection - firstLumiSectionSeen_);
  double fullUpdateRatio = ((double) fullLsDelta) / lumiSectionsPerUpdate_;
  // this is the update number starting from zero
  uint32 updateNumber = -1 + (uint32) fullUpdateRatio;
  // this is the actual luminosity section number for the beginning lumi section of this update
  unsigned int lumiSectionTag = firstLumiSectionSeen_ +
    ((int) (updateNumber * lumiSectionsPerUpdate_));

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
                                     event.id().run(), event.id().event(),
				     event.time(),
                                     lumiSectionTag, updateNumber,
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
    dqmMsgBuilder.setFUProcessId(fuProcId_);
    dqmMsgBuilder.setFUGuid(fuGuidValue_);

    // send the message
    writeShmDQMData(dqmMsgBuilder);

    // test deserialization
    if (DSS_DEBUG >= 3) {
      DQMEventMsgView dqmEventView(&messageBuffer_[0]);
      std::cout << "  DQM Message data:" << std::endl; 
      std::cout << "    protocol version = "
                << dqmEventView.protocolVersion() << std::endl; 
      std::cout << "    header size = "
                << dqmEventView.headerSize() << std::endl; 
      std::cout << "    run number = "
                << dqmEventView.runNumber() << std::endl; 
      std::cout << "    event number = "
                << dqmEventView.eventNumberAtUpdate() << std::endl; 
      std::cout << "    lumi section = "
                << dqmEventView.lumiSection() << std::endl; 
      std::cout << "    update number = "
                << dqmEventView.updateNumber() << std::endl; 
      std::cout << "    compression flag = "
                << dqmEventView.compressionFlag() << std::endl; 
      std::cout << "    reserved word = "
                << dqmEventView.reserved() << std::endl; 
      std::cout << "    release tag = "
                << dqmEventView.releaseTag() << std::endl; 
      std::cout << "    top folder name = "
                << dqmEventView.topFolderName() << std::endl; 
      std::cout << "    sub folder count = "
                << dqmEventView.subFolderCount() << std::endl; 
      std::auto_ptr<DQMEvent::TObjectTable> toTablePtr =
        deserializeWorker_.deserializeDQMEvent(dqmEventView);
      DQMEvent::TObjectTable::const_iterator toIter;
      for (toIter = toTablePtr->begin();
           toIter != toTablePtr->end(); toIter++) {
        std::string subFolderName = toIter->first;
        std::cout << "  folder = " << subFolderName << std::endl;
        std::vector<TObject *> toList = toIter->second;
        for (int tdx = 0; tdx < (int) toList.size(); tdx++) {
          TObject *toPtr = toList[tdx];
          string cls = toPtr->IsA()->GetName();
          string nm = toPtr->GetName();
          std::cout << "    TObject class = " << cls
                    << ", name = " << nm << std::endl;
        }
      }
    }
  }

  // reset monitor elements that have requested it
  // TODO - enable this
  //bei->doneSending(true, true);

  // update the "previous" lumi section
  lumiSectionOfPreviousUpdate_ = thisLumiSection;
}

/**
 * Callback to be used after the input source has been constructed.  It
 * takes care of any intialization that is needed by this service.
 */
void FUShmDQMOutputService::postSourceConstructionProcessing(const edm::ModuleDescription &moduleDesc)
{
  if (DSS_DEBUG) {
    cout << "FUShmDQMOutputService::postSourceConstructionProcessing called for "
         << moduleDesc.moduleName() << endl;
  }

  bei = edm::Service<DQMStore>().operator->();
}

/**
 * Callback to be used after the Run has been created by the InputSource
 * but before any modules have seen the Run
 */
void FUShmDQMOutputService::preBeginRun(const edm::RunID &runID,
                                        const edm::Timestamp &timestamp)
{
  if (DSS_DEBUG) {
    cout << "FUShmDQMOutputService::preBeginRun called, run number "
         << runID.run() << endl;
  }

  initializationIsNeeded_ = true;
}

/**
 * Callback to be used after the end job operation has finished.  It takes
 * care of any necessary cleanup.
 */
void FUShmDQMOutputService::postEndJobProcessing()
{
  if (DSS_DEBUG) {
    cout << "FUShmDQMOutputService::postEndJobProcessing called" << endl;
  }
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
    if (mePtr->wasUpdated()) {
      updateTOList.push_back(mePtr->getRootObject());
    }
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
  if (DSS_DEBUG) {
    std::cout << "Folder = " << topFolder << " crc = " << crc << std::endl;
  }

  if(!shmBuffer_) {
    edm::LogError("FUDQMShmOutputService") 
      << " Error writing to shared memory as shm is not available";
  } else {
    bool ret = shmBuffer_->writeDqmEventData(runid, eventid, (unsigned int)crc,
                                             fuProcId_, fuGuidValue_, buffer, size);
    if(!ret) edm::LogError("FUShmDQMOutputService") << " Error with writing data to ShmBuffer";
  }

}

/**
 * Callback for when the begin job operation is finishing.
 * Currently, this routine is only used for diagnostics.
 */
void FUShmDQMOutputService::postBeginJobProcessing()
{
  if (DSS_DEBUG >= 2) {
    cout << "FUShmDQMOutputService::postBeginJobProcessing called" << endl;
  }
}

/**
 * Callback for when the input source is about to read or generate a
 * physics event.
 * Currently, this routine is only used for diagnostics.
 */
void FUShmDQMOutputService::preSourceProcessing()
{
  if (DSS_DEBUG >= 2) {
    cout << "FUShmDQMOutputService::preSourceProcessing called" << endl;
  }
}

/**
 * Callback for when the input source has finished reading or generating a
 * physics event.
 * Currently, this routine is only used for diagnostics.
 */
void FUShmDQMOutputService::postSourceProcessing()
{
  if (DSS_DEBUG >= 2) {
    cout << "FUShmDQMOutputService::postSourceProcessing called" << endl;
  }
}

/**
 * Callback to be used before an analysis module begins its processing.
 * Currently, this routine is only used for diagnostics.
 */
void FUShmDQMOutputService::preModuleProcessing(const edm::ModuleDescription &moduleDesc)
{
  if (DSS_DEBUG >= 2) {
    cout << "FUShmDQMOutputService::preModuleProcessing called for "
         << moduleDesc.moduleName() << endl;
  }
}

/**
 * Callback to be used after an analysis module has completed its processing.
 * Currently, this routine is only used for diagnostics.
 */
void FUShmDQMOutputService::postModuleProcessing(const edm::ModuleDescription &moduleDesc)
{
  if (DSS_DEBUG >= 2) {
    cout << "FUShmDQMOutputService::postModuleProcessing called for "
         << moduleDesc.moduleName() << endl;
  }
}

/**
 * Callback to be used before the input source is constructed.
 * Currently, this routine is only used for diagnostics.
 */
void FUShmDQMOutputService::preSourceConstructionProcessing(const edm::ModuleDescription &moduleDesc)
{
  if (DSS_DEBUG >= 2) {
    cout << "FUShmDQMOutputService::preSourceConstructionProcessing called for "
         << moduleDesc.moduleName() << endl;
  }
}

/**
 * Callback to be used before analysis modules are constructed.
 * Currently, this routine is only used for diagnostics.
 */
void FUShmDQMOutputService::preModuleConstructionProcessing(const edm::ModuleDescription &moduleDesc)
{
  if (DSS_DEBUG >= 2) {
    cout << "FUShmDQMOutputService::preModuleConstructionProcessing called for "
         << moduleDesc.moduleName() << endl;
  }
}

/**
 * Callback to be used after analysis modules have been constructed.
 * Currently, this routine is only used for diagnostics.
 */
void FUShmDQMOutputService::postModuleConstructionProcessing(const edm::ModuleDescription &moduleDesc)
{
  if (DSS_DEBUG >= 2) {
    cout << "FUShmDQMOutputService::postModuleConstructionProcessing called for "
         << moduleDesc.moduleName() << endl;
  }
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
