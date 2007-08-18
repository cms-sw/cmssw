// $Id: StorageManager.cc,v 1.25 2007/08/15 23:13:54 hcheung Exp $

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <sys/stat.h>

#include "EventFilter/StorageManager/interface/StorageManager.h"
#include "EventFilter/StorageManager/interface/ConsumerPipe.h"
#include "EventFilter/StorageManager/interface/ProgressMarker.h"
#include "EventFilter/StorageManager/interface/Configurator.h"
#include "EventFilter/StorageManager/interface/Parameter.h"
#include "EventFilter/StorageManager/interface/FUProxy.h"

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
#include "EventFilter/Utilities/interface/ParameterSetRetriever.h"

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include "IOPool/Streamer/interface/MsgHeader.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/OtherMessage.h"
#include "IOPool/Streamer/interface/ConsRegMessage.h"
#include "IOPool/Streamer/interface/HLTInfo.h"
#include "IOPool/Streamer/interface/Utilities.h"
#include "IOPool/Streamer/interface/TestFileReader.h"
#include "IOPool/Streamer/interface/StreamerInputSource.h"

#include "xcept/tools.h"

#include "i2o/Method.h"
#include "i2o/utils/AddressMap.h"

#include "toolbox/mem/Pool.h"

#include "xcept/tools.h"

#include "xgi/Method.h"

#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPBody.h"
#include "xoap/domutils.h"

#include "xdata/InfoSpaceFactory.h"

namespace stor {
extern bool getSMFC_exceptionStatus();
extern std::string getSMFC_reason4Exception();
}

using namespace edm;
using namespace std;
using namespace stor;


static void deleteSMBuffer(void* Ref)
{
  // release the memory pool buffer
  // once the fragment collector is done with it
  stor::FragEntry* entry = (stor::FragEntry*)Ref;
  // check code for INIT message 
  // and all messages work like this (going into a queue)
  // do not delete the memory for the single (first) INIT message
  // it is stored in the local data member for event server
  // but should not keep all INIT messages? Clean this up!
  if(entry->code_ != Header::INIT) 
  {
    toolbox::mem::Reference *ref=(toolbox::mem::Reference*)entry->buffer_object_;
    ref->release();
  }
}


StorageManager::StorageManager(xdaq::ApplicationStub * s)
  throw (xdaq::exception::Exception) :
  xdaq::Application(s),
  fsm_(this), 
  reasonForFailedState_(),
  ah_(0), 
  pushMode_(false), 
  collateDQM_(false),
  archiveDQM_(false),
  filePrefixDQM_("/tmp/DQM"),
  purgeTimeDQM_(DEFAULT_PURGE_TIME),
  readyTimeDQM_(DEFAULT_READY_TIME),
  useCompressionDQM_(true),
  compressionLevelDQM_(1),
  serialized_prods_(1000000),
  ser_prods_size_(0),
  mybuffer_(7000000),
  connectedFUs_(0), 
  storedEvents_(0), 
  dqmRecords_(0), 
  storedVolume_(0.),
  progressMarker_(ProgressMarker::instance()->idle())
{  
  LOG4CPLUS_INFO(this->getApplicationLogger(),"Making StorageManager");

  ah_   = new edm::AssertHandler();
  fsm_.initialize<StorageManager>(this);

  // Careful with next line: state machine fsm_ has to be setup first
  setupFlashList();

  xdata::InfoSpace *ispace = getApplicationInfoSpace();

  ispace->fireItemAvailable("STparameterSet",&offConfig_);
  ispace->fireItemAvailable("runNumber",     &runNumber_);
  ispace->fireItemAvailable("stateName",     fsm_.stateName());
  ispace->fireItemAvailable("connectedFUs",  &connectedFUs_);
  ispace->fireItemAvailable("storedEvents",  &storedEvents_);
  ispace->fireItemAvailable("dqmRecords",    &dqmRecords_);
  ispace->fireItemAvailable("closedFiles",&closedFiles_);
  ispace->fireItemAvailable("fileList",&fileList_);
  ispace->fireItemAvailable("eventsInFile",&eventsInFile_);
  ispace->fireItemAvailable("fileSize",&fileSize_);

  // Bind specific messages to functions
  i2o::bind(this,
            &StorageManager::receiveRegistryMessage,
            I2O_SM_PREAMBLE,
            XDAQ_ORGANIZATION_ID);
  i2o::bind(this,
            &StorageManager::receiveDataMessage,
            I2O_SM_DATA,
            XDAQ_ORGANIZATION_ID);
  i2o::bind(this,
            &StorageManager::receiveOtherMessage,
            I2O_SM_OTHER,
            XDAQ_ORGANIZATION_ID);
  i2o::bind(this,
            &StorageManager::receiveDQMMessage,
            I2O_SM_DQM,
            XDAQ_ORGANIZATION_ID);

  // Bind web interface
  xgi::bind(this,&StorageManager::defaultWebPage,       "Default");
  xgi::bind(this,&StorageManager::css,                  "styles.css");
  xgi::bind(this,&StorageManager::fusenderWebPage,      "fusenderlist");
  xgi::bind(this,&StorageManager::streamerOutputWebPage,"streameroutput");
  xgi::bind(this,&StorageManager::eventdataWebPage,     "geteventdata");
  xgi::bind(this,&StorageManager::headerdataWebPage,    "getregdata");
  xgi::bind(this,&StorageManager::consumerWebPage,      "registerConsumer");
  xgi::bind(this,&StorageManager::DQMeventdataWebPage,     "getDQMeventdata");
  xgi::bind(this,&StorageManager::DQMconsumerWebPage,      "registerDQMConsumer");
  receivedFrames_ = 0;
  pool_is_set_    = 0;
  pool_           = 0;
  nLogicalDisk_   = 0;
  pushmode2proxy_ = false;

  // Variables needed for streamer file writing
  ispace->fireItemAvailable("pushMode2Proxy", &pushmode2proxy_);
  ispace->fireItemAvailable("collateDQM",     &collateDQM_);
  ispace->fireItemAvailable("archiveDQM",     &archiveDQM_);
  ispace->fireItemAvailable("purgeTimeDQM",   &purgeTimeDQM_);
  ispace->fireItemAvailable("readyTimeDQM",   &readyTimeDQM_);
  ispace->fireItemAvailable("filePrefixDQM",  &filePrefixDQM_);
  ispace->fireItemAvailable("useCompressionDQM",  &useCompressionDQM_);
  ispace->fireItemAvailable("compressionLevelDQM",  &compressionLevelDQM_);
  ispace->fireItemAvailable("nLogicalDisk", &nLogicalDisk_);

  boost::shared_ptr<stor::Parameter> smParameter_ = stor::Configurator::instance()->getParameter();
  closeFileScript_    = smParameter_ -> closeFileScript();
  notifyTier0Script_  = smParameter_ -> notifyTier0Script();
  insertFileScript_   = smParameter_ -> insertFileScript();  
  fileCatalog_        = smParameter_ -> fileCatalog(); 
  fileName_           = smParameter_ -> fileName();
  filePath_           = smParameter_ -> filePath();
  mailboxPath_        = smParameter_ -> mailboxPath();
  setupLabel_         = smParameter_ -> setupLabel();
  highWaterMark_      = smParameter_ -> highWaterMark();
  lumiSectionTimeOut_ = smParameter_ -> lumiSectionTimeOut();

  ispace->fireItemAvailable("closeFileScript",    &closeFileScript_);
  ispace->fireItemAvailable("notifyTier0Script",  &notifyTier0Script_);
  ispace->fireItemAvailable("insertFileScript",   &insertFileScript_);
  ispace->fireItemAvailable("fileCatalog",        &fileCatalog_);
  ispace->fireItemAvailable("fileName",           &fileName_);
  ispace->fireItemAvailable("filePath",           &filePath_);
  ispace->fireItemAvailable("mailboxPath",        &mailboxPath_);
  ispace->fireItemAvailable("setupLabel",         &setupLabel_);
  ispace->fireItemAvailable("highWaterMark",      &highWaterMark_);
  ispace->fireItemAvailable("lumiSectionTimeOut", &lumiSectionTimeOut_);

  // added for Event Server
  ser_prods_size_ = 0;
  serialized_prods_[0] = '\0';
  maxESEventRate_ = 1.0;  // hertz
  ispace->fireItemAvailable("maxESEventRate",&maxESEventRate_);
  activeConsumerTimeout_ = 300;  // seconds
  ispace->fireItemAvailable("activeConsumerTimeout",&activeConsumerTimeout_);
  idleConsumerTimeout_ = 600;  // seconds
  ispace->fireItemAvailable("idleConsumerTimeout",&idleConsumerTimeout_);
  consumerQueueSize_ = 5;
  ispace->fireItemAvailable("consumerQueueSize",&consumerQueueSize_);
  DQMmaxESEventRate_ = 1.0;  // hertz
  ispace->fireItemAvailable("DQMmaxESEventRate",&DQMmaxESEventRate_);
  DQMactiveConsumerTimeout_ = 300;  // seconds
  ispace->fireItemAvailable("DQMactiveConsumerTimeout",&DQMactiveConsumerTimeout_);
  DQMidleConsumerTimeout_ = 600;  // seconds
  ispace->fireItemAvailable("DQMidleConsumerTimeout",&DQMidleConsumerTimeout_);
  DQMconsumerQueueSize_ = 5;
  ispace->fireItemAvailable("DQMconsumerQueueSize",&DQMconsumerQueueSize_);

  // for performance measurements
  samples_          = 100; // measurements every 25MB (about)
  instantBandwidth_ = 0.;
  instantRate_      = 0.;
  instantLatency_   = 0.;
  totalSamples_     = 0;
  duration_         = 0.;
  meanBandwidth_    = 0.;
  meanRate_         = 0.;
  meanLatency_      = 0.;
  maxBandwidth_     = 0.;
  minBandwidth_     = 999999.;

  pmeter_ = new stor::SMPerformanceMeter();
  pmeter_->init(samples_);

  string        xmlClass = getApplicationDescriptor()->getClassName();
  unsigned long instance = getApplicationDescriptor()->getInstance();
  ostringstream sourcename;
  // sourcename << xmlClass << "_" << instance;
  sourcename << instance;
  sourceId_ = sourcename.str();
  smParameter_ -> setSmInstance(sourceId_);  // sourceId_ can be removed ...

  // need the line below so that deserializeRegistry can run
  // in order to compare two registries (cannot compare byte-for-byte) (if we keep this)
  // need line below anyway in case we deserialize DQMEvents for collation
  edm::RootAutoLibraryLoader::enable();
}

StorageManager::~StorageManager()
{
  delete ah_;
  delete pmeter_;
}

xoap::MessageReference
StorageManager::ParameterGet(xoap::MessageReference message)
  throw (xoap::exception::Exception)
{
  connectedFUs_.value_ = smfusenders_.size();
  return Application::ParameterGet(message);
}




////////// *** I2O frame call back functions /////////////////////////////////////////////
void StorageManager::receiveRegistryMessage(toolbox::mem::Reference *ref)
{
  // get the memory pool pointer for statistics if not already set
  if(pool_is_set_ == 0)
  {
    pool_ = ref->getBuffer()->getPool();
    pool_is_set_ = 1;
  }

  I2O_MESSAGE_FRAME         *stdMsg  = (I2O_MESSAGE_FRAME*) ref->getDataLocation();
  I2O_SM_PREAMBLE_MESSAGE_FRAME *msg = (I2O_SM_PREAMBLE_MESSAGE_FRAME*) stdMsg;

  FDEBUG(10) << "StorageManager: Received registry message from HLT " << msg->hltURL
             << " application " << msg->hltClassName << " id " << msg->hltLocalId
             << " instance " << msg->hltInstance << " tid " << msg->hltTid << std::endl;
  FDEBUG(10) << "StorageManager: registry size " << msg->dataSize << "\n";

  // *** check the Storage Manager is in the Ready or Enabled state first!
  if(fsm_.stateName()->toString() != "Enabled" && fsm_.stateName()->toString() != "Ready" )
  {
    LOG4CPLUS_ERROR(this->getApplicationLogger(),
                       "Received INIT message but not in Ready/Enabled state! Current state = "
                       << fsm_.stateName()->toString() << " INIT from " << msg->hltURL
                       << " application " << msg->hltClassName);
    // just release the memory at least - is that what we want to do?
    ref->release();
    return;
  }
  receivedFrames_++;

  // for bandwidth performance measurements
  unsigned long actualFrameSize = (unsigned long)sizeof(I2O_SM_PREAMBLE_MESSAGE_FRAME)
                                  + msg->dataSize;
  addMeasurement(actualFrameSize);
  // register this FU sender into the list to keep its status
  int status = smfusenders_.registerFUSender(&msg->hltURL[0], &msg->hltClassName[0],
                 msg->hltLocalId, msg->hltInstance, msg->hltTid,
                 msg->frameCount, msg->numFrames, ref);
  // see if this completes the registry data for this FU
  // if so then: if first copy it, if subsequent test it (mark which is first?)
  // should test on -1 as problems
  if(status == 1)
  {
    char* regPtr = smfusenders_.getRegistryData(&msg->hltURL[0], &msg->hltClassName[0],
                 msg->hltLocalId, msg->hltInstance, msg->hltTid);
    unsigned int regSz = smfusenders_.getRegistrySize(&msg->hltURL[0], &msg->hltClassName[0],
                 msg->hltLocalId, msg->hltInstance, msg->hltTid);
    // Assume first received registry is correct and save it
    if(ser_prods_size_ == 0)
    {
      if(serialized_prods_.capacity() < regSz) serialized_prods_.resize(regSz);
      copy(regPtr, regPtr+regSz, &serialized_prods_[0]);
      ser_prods_size_ = regSz;
      FDEBUG(9) << "Saved serialized registry for Event Server, size "
                << ser_prods_size_ << std::endl;
      // queue for output
      EventBuffer::ProducerBuffer b(jc_->getFragmentQueue());
      // don't have the correct run number yet
      new (b.buffer()) stor::FragEntry(&serialized_prods_[0], &serialized_prods_[0], ser_prods_size_,
                                       1, 1, Header::INIT, 0, 0, 0); // use fixed 0 as ID
      b.commit(sizeof(stor::FragEntry));
      // this is checked ok by default
      smfusenders_.setRegCheckedOK(&msg->hltURL[0], &msg->hltClassName[0],
                                   msg->hltLocalId, msg->hltInstance, msg->hltTid);
    } else { // this is the second or subsequent INIT message test it against first
      // cannot test byte for byte the serialized registry
      InitMsgView testmsg(regPtr);
      InitMsgView refmsg(&serialized_prods_[0]);
      std::auto_ptr<edm::SendJobHeader> header = StreamerInputSource::deserializeRegistry(testmsg);
      std::auto_ptr<edm::SendJobHeader> refheader = StreamerInputSource::deserializeRegistry(refmsg);
      // should test this well to see if a try block is needed for next line
      if(edm::registryIsSubset(*header, *refheader)) {  // using clunky method
        FDEBUG(9) << "copyAndTestRegistry: Received registry is okay" << std::endl;
        smfusenders_.setRegCheckedOK(&msg->hltURL[0], &msg->hltClassName[0],
                                     msg->hltLocalId, msg->hltInstance, msg->hltTid);
      } else {
        // should do something with subsequent data from this FU!
        FDEBUG(9) << "copyAndTestRegistry: Error! Received registry is not a subset!"
                  << " for URL " << msg->hltURL << " and Tid " << msg->hltTid
                  << std::endl;
        LOG4CPLUS_ERROR(this->getApplicationLogger(),
                        "copyAndTestRegistry: Error! Received registry is not a subset!"
                        << " for URL " << msg->hltURL << " and Tid " << msg->hltTid);
      }
      // end of streamer writing test - should an else with an error/warning message
      // or decide once and for all we only write streamer files and get rid of test
    } // end of test on if registry data was saved

    string hltClassName(msg->hltClassName);
    sendDiscardMessage(msg->fuID, 
		       msg->hltInstance, 
		       I2O_FU_DATA_DISCARD,
		       hltClassName);
  } // end of test on if registryFUSender returned that registry is complete
}

void StorageManager::receiveDataMessage(toolbox::mem::Reference *ref)
{
  // get the memory pool pointer for statistics if not already set
  if(pool_is_set_ == 0)
  {
    pool_ = ref->getBuffer()->getPool();
    pool_is_set_ = 1;
  }

  I2O_MESSAGE_FRAME         *stdMsg =
    (I2O_MESSAGE_FRAME*)ref->getDataLocation();
  I2O_SM_DATA_MESSAGE_FRAME *msg    =
    (I2O_SM_DATA_MESSAGE_FRAME*)stdMsg;
  FDEBUG(10)   << "StorageManager: Received data message from HLT at " << msg->hltURL 
	       << " application " << msg->hltClassName << " id " << msg->hltLocalId
	       << " instance " << msg->hltInstance << " tid " << msg->hltTid << std::endl;
  FDEBUG(10)   << "                 for run " << msg->runID << " event " << msg->eventID
	       << " total frames = " << msg->numFrames << std::endl;
  FDEBUG(10)   << "StorageManager: Frame " << msg->frameCount << " of " 
	       << msg->numFrames-1 << std::endl;
  
  int len = msg->dataSize;

  // check the storage Manager is in the Ready state first!
  if(fsm_.stateName()->toString() != "Enabled")
  {
    LOG4CPLUS_ERROR(this->getApplicationLogger(),
                       "Received EVENT message but not in Enabled state! Current state = "
                       << fsm_.stateName()->toString() << " EVENT from" << msg->hltURL
                       << " application " << msg->hltClassName);
    // just release the memory at least - is that what we want to do?
    ref->release();
    return;
  }

  // If running with local transfers, a chain of I2O frames when posted only has the
  // head frame sent. So a single frame can complete a chain for local transfers.
  // We need to test for this. Must be head frame, more than one frame
  // and next pointer must exist.
  int is_local_chain = 0;
  if(msg->frameCount == 0 && msg->numFrames > 1 && ref->getNextReference())
  {
    // this looks like a chain of frames (local transfer)
    toolbox::mem::Reference *head = ref;
    toolbox::mem::Reference *next = 0;
    // best to check the complete chain just in case!
    unsigned int tested_frames = 1;
    next = head;
    while((next=next->getNextReference())!=0) tested_frames++;
    FDEBUG(10) << "StorageManager: Head frame has " << tested_frames-1
               << " linked frames out of " << msg->numFrames-1 << std::endl;
    if(msg->numFrames == tested_frames)
    {
      // found a complete linked chain from the leading frame
      is_local_chain = 1;
      FDEBUG(10) << "StorageManager: Leading frame contains a complete linked chain"
                 << " - must be local transfer" << std::endl;
      FDEBUG(10) << "StorageManager: Breaking the chain" << std::endl;
      // break the chain and feed them to the fragment collector
      next = head;

      for(int iframe=0; iframe <(int)msg->numFrames; iframe++)
      {
         toolbox::mem::Reference *thisref=next;
         next = thisref->getNextReference();
         thisref->setNextReference(0);
         I2O_MESSAGE_FRAME         *thisstdMsg = (I2O_MESSAGE_FRAME*)thisref->getDataLocation();
         I2O_SM_DATA_MESSAGE_FRAME *thismsg    = (I2O_SM_DATA_MESSAGE_FRAME*)thisstdMsg;
         EventBuffer::ProducerBuffer b(jc_->getFragmentQueue());
         int thislen = thismsg->dataSize;
         // ***  must give it the 1 of N for this fragment (starts from 0 in i2o header)
         new (b.buffer()) stor::FragEntry(thisref, (char*)(thismsg->dataPtr()), thislen,
                  thismsg->frameCount+1, thismsg->numFrames, Header::EVENT, 
                  thismsg->runID, thismsg->eventID, 0);
         b.commit(sizeof(stor::FragEntry));

         receivedFrames_++;
         // for bandwidth performance measurements
         // Following is wrong for the last frame because frame sent is
         // is actually larger than the size taken by actual data
         unsigned long actualFrameSize = (unsigned long)sizeof(I2O_SM_DATA_MESSAGE_FRAME)
                                         +thislen;
         addMeasurement(actualFrameSize);

         // should only do this test if the first data frame from each FU?
         // check if run number is the same as that in Run configuration, complain otherwise !!!
         // this->runNumber_ comes from the RunBase class that StorageManager inherits from
         if(msg->runID != runNumber_)
         {
           LOG4CPLUS_ERROR(this->getApplicationLogger(),"Run Number from event stream = " << msg->runID
                           << " From " << msg->hltURL
                           << " Different from Run Number from configuration = " << runNumber_);
         }
         // for FU sender list update
         // msg->frameCount start from 0, but in EventMsg header it starts from 1!
         bool isLocal = true;

         int status = 
	   smfusenders_.updateFUSender4data(&msg->hltURL[0], &msg->hltClassName[0],
					    msg->hltLocalId, msg->hltInstance, msg->hltTid,
					    msg->runID, msg->eventID, msg->frameCount+1, msg->numFrames,
					    msg->originalSize, isLocal);

         if(status == 1) ++(storedEvents_.value_);
         if(status == -1) {
           LOG4CPLUS_ERROR(this->getApplicationLogger(),
                    "updateFUSender4data: Cannot find FU in FU Sender list!"
                    << " With URL "
                    << msg->hltURL << " class " << msg->hltClassName  << " instance "
                    << msg->hltInstance << " Tid " << msg->hltTid);
         }
      }

    } else {
      // should never get here!
      FDEBUG(10) << "StorageManager: Head frame has fewer linked frames "
                 << "than expected: abnormal error! " << std::endl;
    }
  }

  if (is_local_chain == 0) 
  {
    // put pointers into fragment collector queue
    EventBuffer::ProducerBuffer b(jc_->getFragmentQueue());
    // must give it the 1 of N for this fragment (starts from 0 in i2o header)
    /* stor::FragEntry* fe = */ new (b.buffer()) stor::FragEntry(ref, (char*)(msg->dataPtr()), len,
                                msg->frameCount+1, msg->numFrames, Header::EVENT, 
                                msg->runID, msg->eventID, 0);
    b.commit(sizeof(stor::FragEntry));
    // Frame release is done in the deleter.
    receivedFrames_++;
    // for bandwidth performance measurements
    unsigned long actualFrameSize = (unsigned long)sizeof(I2O_SM_DATA_MESSAGE_FRAME)
                                    + len;
    addMeasurement(actualFrameSize);

    // should only do this test if the first data frame from each FU?
    // check if run number is the same as that in Run configuration, complain otherwise !!!
    // this->runNumber_ comes from the RunBase class that StorageManager inherits from
    if(msg->runID != runNumber_)
    {
      LOG4CPLUS_ERROR(this->getApplicationLogger(),"Run Number from event stream = " << msg->runID
                      << " From " << msg->hltURL
                      << " Different from Run Number from configuration = " << runNumber_);
    }
    // for FU sender list update
    // msg->frameCount start from 0, but in EventMsg header it starts from 1!
    bool isLocal = false;
    int status = 
      smfusenders_.updateFUSender4data(&msg->hltURL[0], &msg->hltClassName[0],
				       msg->hltLocalId, msg->hltInstance, msg->hltTid,
				       msg->runID, msg->eventID, msg->frameCount+1, msg->numFrames,
				       msg->originalSize, isLocal);
    
    if(status == 1) ++(storedEvents_.value_);
    if(status == -1) {
      LOG4CPLUS_ERROR(this->getApplicationLogger(),
		      "updateFUSender4data: Cannot find FU in FU Sender list!"
		      << " With URL "
		      << msg->hltURL << " class " << msg->hltClassName  << " instance "
		      << msg->hltInstance << " Tid " << msg->hltTid);
    }
  }

  if (  msg->frameCount == msg->numFrames-1 )
    {
      string hltClassName(msg->hltClassName);
      sendDiscardMessage(msg->fuID, 
			 msg->hltInstance, 
			 I2O_FU_DATA_DISCARD,
			 hltClassName);
    }
}


void StorageManager::receiveOtherMessage(toolbox::mem::Reference *ref)
{
  // get the memory pool pointer for statistics if not already set
  if(pool_is_set_ == 0)
  {
    pool_ = ref->getBuffer()->getPool();
   pool_is_set_ = 1;
  }

  I2O_MESSAGE_FRAME         *stdMsg = (I2O_MESSAGE_FRAME*)ref->getDataLocation();
  I2O_SM_OTHER_MESSAGE_FRAME *msg    = (I2O_SM_OTHER_MESSAGE_FRAME*)stdMsg;
  FDEBUG(9) << "StorageManager: Received other message from HLT " << msg->hltURL
             << " application " << msg->hltClassName << " id " << msg->hltLocalId
             << " instance " << msg->hltInstance << " tid " << msg->hltTid << std::endl;
  FDEBUG(9) << "StorageManager: message content " << msg->otherData << "\n";
 
  // Not yet processing any Other messages type
  // the only "other" message is an end-of-run. It is awaited to process a request to halt the storage manager

  // check the storage Manager is in the correct state to process each message
  // the end-of-run message is only valid when in the "Enabled" state
  if(fsm_.stateName()->toString() != "Enabled")
  {
    LOG4CPLUS_ERROR(this->getApplicationLogger(),
                       "Received OTHER (End-of-run) message but not in Enabled state! Current state = "
                       << fsm_.stateName()->toString() << " OTHER from" << msg->hltURL
                       << " application " << msg->hltClassName);
    // just release the memory at least - is that what we want to do?
    ref->release();
    return;
  }
  
  LOG4CPLUS_INFO(this->getApplicationLogger(),"removing FU sender at " << msg->hltURL);
  bool didErase = smfusenders_.removeFUSender(&msg->hltURL[0], &msg->hltClassName[0],
		 msg->hltLocalId, msg->hltInstance, msg->hltTid);
  if(!didErase)
  {
    LOG4CPLUS_ERROR(this->getApplicationLogger(),
                    "Spurious end-of-run received for FU not in Sender list!"
                    << " With URL "
                    << msg->hltURL << " class " << msg->hltClassName  << " instance "
                    << msg->hltInstance << " Tid " << msg->hltTid);
  }
  
  // release the frame buffer now that we are finished
  ref->release();

  receivedFrames_++;

  // for bandwidth performance measurements
  unsigned long actualFrameSize = (unsigned long)sizeof(I2O_SM_OTHER_MESSAGE_FRAME);
  addMeasurement(actualFrameSize);
}

void StorageManager::receiveDQMMessage(toolbox::mem::Reference *ref)
{
  // get the memory pool pointer for statistics if not already set
  if(pool_is_set_ == 0)
  {
    pool_ = ref->getBuffer()->getPool();
    pool_is_set_ = 1;
  }

  I2O_MESSAGE_FRAME         *stdMsg =
    (I2O_MESSAGE_FRAME*)ref->getDataLocation();
  I2O_SM_DQM_MESSAGE_FRAME *msg    =
    (I2O_SM_DQM_MESSAGE_FRAME*)stdMsg;
  FDEBUG(10) << "StorageManager: Received DQM message from HLT at " << msg->hltURL 
             << " application " << msg->hltClassName << " id " << msg->hltLocalId
             << " instance " << msg->hltInstance << " tid " << msg->hltTid << std::endl;
  FDEBUG(10) << "                 for run " << msg->runID << " eventATUpdate = " << msg->eventAtUpdateID
             << " total frames = " << msg->numFrames << std::endl;
  FDEBUG(10) << "StorageManager: Frame " << msg->frameCount << " of " 
             << msg->numFrames-1 << std::endl;
  int len = msg->dataSize;
  FDEBUG(10) << "StorageManager: received DQM frame size = " << len << std::endl;

  // check the storage Manager is in the Ready state first!
  if(fsm_.stateName()->toString() != "Enabled")
  {
    LOG4CPLUS_ERROR(this->getApplicationLogger(),
                       "Received DQM message but not in Enabled state! Current state = "
                       << fsm_.stateName()->toString() << " DQMMessage from" << msg->hltURL
                       << " application " << msg->hltClassName);
    // just release the memory at least - is that what we want to do?
    ref->release();
    return;
  }
  (dqmRecords_.value_)++;

  // If running with local transfers, a chain of I2O frames when posted only has the
  // head frame sent. So a single frame can complete a chain for local transfers.
  // We need to test for this. Must be head frame, more than one frame
  // and next pointer must exist.
  // -- we have to break chains due to the way the FragmentCollector frees memory
  //    for each frame after processing each as the freeing a chain frees all memory
  //    in the chain
  int is_local_chain = 0;
  if(msg->frameCount == 0 && msg->numFrames > 1 && ref->getNextReference())
  {
    // this looks like a chain of frames (local transfer)
    toolbox::mem::Reference *head = ref;
    toolbox::mem::Reference *next = 0;
    // best to check the complete chain just in case!
    unsigned int tested_frames = 1;
    next = head;
    while((next=next->getNextReference())!=0) ++tested_frames;
    FDEBUG(10) << "StorageManager: DQM Head frame has " << tested_frames-1
               << " linked frames out of " << msg->numFrames-1 << std::endl;
    if(msg->numFrames == tested_frames)
    {
      // found a complete linked chain from the leading frame
      is_local_chain = 1;
      FDEBUG(10) << "StorageManager: Leading frame contains a complete linked chain"
                 << " - must be local transfer" << std::endl;
      FDEBUG(10) << "StorageManager: Breaking the chain" << std::endl;
      // break the chain and feed them to the fragment collector
      next = head;

      for(int iframe=0; iframe <(int)msg->numFrames; ++iframe)
      {
         toolbox::mem::Reference *thisref=next;
         next = thisref->getNextReference();
         thisref->setNextReference(0);
         I2O_MESSAGE_FRAME         *thisstdMsg = (I2O_MESSAGE_FRAME*)thisref->getDataLocation();
         I2O_SM_DQM_MESSAGE_FRAME *thismsg    = (I2O_SM_DQM_MESSAGE_FRAME*)thisstdMsg;
         EventBuffer::ProducerBuffer b(jc_->getFragmentQueue());
         int thislen = thismsg->dataSize;
         // ***  must give it the 1 of N for this fragment (starts from 0 in i2o header)
         new (b.buffer()) stor::FragEntry(thisref, (char*)(thismsg->dataPtr()), thislen,
                  thismsg->frameCount+1, thismsg->numFrames, Header::DQM_EVENT, 
                  thismsg->runID, thismsg->eventAtUpdateID, thismsg->folderID);
         b.commit(sizeof(stor::FragEntry));

         ++receivedFrames_;
         // for bandwidth performance measurements
         // Following is wrong for the last frame because frame sent is
         // is actually larger than the size taken by actual data
         unsigned long actualFrameSize = (unsigned long)sizeof(I2O_SM_DQM_MESSAGE_FRAME)
                                         +thislen;
         addMeasurement(actualFrameSize);

         // no FU sender list update yet for DQM data, should add it here
      }

    } else {
      // should never get here!
      FDEBUG(10) << "StorageManager: DQM Head frame has fewer linked frames "
                 << "than expected: abnormal error! " << std::endl;
      LOG4CPLUS_ERROR(this->getApplicationLogger(),"DQM Head frame has fewer linked frames" 
                      << " than expected: abnormal error! ");
    }
  }

  if (is_local_chain == 0) 
  {
    // put pointers into fragment collector queue
    EventBuffer::ProducerBuffer b(jc_->getFragmentQueue());
    // must give it the 1 of N for this fragment (starts from 0 in i2o header)
    /* stor::FragEntry* fe = */ new (b.buffer()) stor::FragEntry(ref, (char*)(msg->dataPtr()), len,
                                msg->frameCount+1, msg->numFrames, Header::DQM_EVENT, 
                                msg->runID, msg->eventAtUpdateID, msg->folderID);
    b.commit(sizeof(stor::FragEntry));
    // Frame release is done in the deleter.
    ++receivedFrames_;
    // for bandwidth performance measurements
    unsigned long actualFrameSize = (unsigned long)sizeof(I2O_SM_DQM_MESSAGE_FRAME)
                                    + len;
    addMeasurement(actualFrameSize);

    // no FU sender list update yet for DQM data, should add it here
  }

  if (  msg->frameCount == msg->numFrames-1 )
    {
      string hltClassName(msg->hltClassName);
      sendDiscardMessage(msg->fuID, 
			 msg->hltInstance, 
			 I2O_FU_DQM_DISCARD,
			 hltClassName);
    }
}
//////////// ***  Performance //////////////////////////////////////////////////////////
void StorageManager::addMeasurement(unsigned long size)
{
  // for bandwidth performance measurements
  if ( pmeter_->addSample(size) )
  {
    // don't print out info by default (already in web page)
    //LOG4CPLUS_INFO(getApplicationLogger(),
    //  toolbox::toString("measured latency: %f for size %d",pmeter_->latency(), size));
    //LOG4CPLUS_INFO(getApplicationLogger(),
    //  toolbox::toString("latency:  %f, rate: %f,bandwidth %f, size: %d\n",
    //  pmeter_->latency(),pmeter_->rate(),pmeter_->bandwidth(),size));
    // new measurement; so update

    // Copy measurements for our record
    instantBandwidth_ = pmeter_->bandwidth();
    instantRate_      = pmeter_->rate();
    instantLatency_   = pmeter_->latency();
    totalSamples_     = pmeter_->totalsamples();
    duration_         = pmeter_->duration();
    meanBandwidth_    = pmeter_->meanbandwidth();
    meanRate_         = pmeter_->meanrate();
    meanLatency_      = pmeter_->meanlatency();

    // Determine minimum and maximum instantaneous bandwidth
    if (instantBandwidth_ > maxBandwidth_)
      maxBandwidth_ = instantBandwidth_;
    if (instantBandwidth_ < minBandwidth_)
      minBandwidth_ = instantBandwidth_;
  }

  // TODO fixme: Find a better place to put this testing of the Fragment Collector thread status!
  // leave this for now until we have the transition available and have clean up code
  if(stor::getSMFC_exceptionStatus()) {
    // there was a fatal exception in the Fragmentation Collector and
    // we want to go to a fail state
    //reasonForFailedState_  = stor::getSMFC_reason4Exception();
    //fsm_.fireFailed(reasonForFailedState_,this);
    edm::LogError("StorageManager") << "Fatal problem in FragmentCollector thread detected! \n"
       << stor::getSMFC_reason4Exception();
  }
}

//////////// *** Default web page //////////////////////////////////////////////////////////
void StorageManager::defaultWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  *out << "<html>"                                                   << endl;
  *out << "<head>"                                                   << endl;
  *out << "<link type=\"text/css\" rel=\"stylesheet\"";
  *out << " href=\"/" <<  getApplicationDescriptor()->getURN()
       << "/styles.css\"/>"                   << endl;
  *out << "<title>" << getApplicationDescriptor()->getClassName() << " instance "
       << getApplicationDescriptor()->getInstance()
       << "</title>"     << endl;
    *out << "<table border=\"0\" width=\"100%\">"                      << endl;
    *out << "<tr>"                                                     << endl;
    *out << "  <td align=\"left\">"                                    << endl;
    *out << "    <img"                                                 << endl;
    *out << "     align=\"middle\""                                    << endl;
    *out << "     src=\"/daq/evb/examples/fu/images/fu64x64.gif\""     << endl;
    *out << "     alt=\"main\""                                        << endl;
    *out << "     width=\"64\""                                        << endl;
    *out << "     height=\"64\""                                       << endl;
    *out << "     border=\"\"/>"                                       << endl;
    *out << "    <b>"                                                  << endl;
    *out << getApplicationDescriptor()->getClassName() << " instance "
         << getApplicationDescriptor()->getInstance()                  << endl;
    *out << "      " << fsm_.stateName()->toString()                   << endl;
    *out << "    </b>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "    <a href=\"/urn:xdaq-application:lid=3\">"             << endl;
    *out << "      <img"                                               << endl;
    *out << "       align=\"middle\""                                  << endl;
    *out << "       src=\"/daq/xdaq/hyperdaq/images/HyperDAQ.jpg\""    << endl;
    *out << "       alt=\"HyperDAQ\""                                  << endl;
    *out << "       width=\"32\""                                      << endl;
    *out << "       height=\"32\""                                      << endl;
    *out << "       border=\"\"/>"                                     << endl;
    *out << "    </a>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "    <a href=\"/" << getApplicationDescriptor()->getURN()
         << "/debug\">"                   << endl;
    *out << "      <img"                                               << endl;
    *out << "       align=\"middle\""                                  << endl;
    *out << "       src=\"/daq/evb/bu/images/debug32x32.gif\""         << endl;
    *out << "       alt=\"debug\""                                     << endl;
    *out << "       width=\"32\""                                      << endl;
    *out << "       height=\"32\""                                     << endl;
    *out << "       border=\"\"/>"                                     << endl;
    *out << "    </a>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "</tr>"                                                    << endl;
    if(fsm_.stateName()->value_ == "Failed")
    {
      *out << "<tr>"					     << endl;
      *out << " <td>"					     << endl;
      *out << "<textarea rows=" << 5 << " cols=60 scroll=yes";
      *out << " readonly title=\"Reason For Failed\">"		     << endl;
      *out << reasonForFailedState_                                  << endl;
      *out << "</textarea>"                                          << endl;
      *out << " </td>"					     << endl;
      *out << "</tr>"					     << endl;
    }
    *out << "</table>"                                                 << endl;

  *out << "<hr/>"                                                    << endl;
  *out << "<table>"                                                  << endl;
  *out << "<tr valign=\"top\">"                                      << endl;
  *out << "  <td>"                                                   << endl;

  *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">" << endl;
  *out << "<colgroup> <colgroup align=\"rigth\">"                    << endl;
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Memory Pool Usage"                            << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;

        *out << "<tr>" << endl;
        *out << "<th >" << endl;
        *out << "Parameter" << endl;
        *out << "</th>" << endl;
        *out << "<th>" << endl;
        *out << "Value" << endl;
        *out << "</th>" << endl;
        *out << "</tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Frames Received" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << receivedFrames_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Events Received" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << storedEvents_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "DQM Records Received" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << dqmRecords_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        if(pool_is_set_ == 1) 
        {
          *out << "<tr>" << endl;
            *out << "<td >" << endl;
            *out << "Memory Used (Bytes)" << endl;
            *out << "</td>" << endl;
            *out << "<td align=right>" << endl;
            *out << pool_->getMemoryUsage().getUsed() << endl;
            *out << "</td>" << endl;
          *out << "  </tr>" << endl;
        } else {
          *out << "<tr>" << endl;
            *out << "<td >" << endl;
            *out << "Memory Pool pointer not yet available" << endl;
            *out << "</td>" << endl;
          *out << "  </tr>" << endl;
        }
// performance statistics
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Performance for last " << samples_ << " frames"<< endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << instantBandwidth_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Rate (Frames/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << instantRate_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Latency (us/frame)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << instantLatency_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Maximum Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << maxBandwidth_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Minimum Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << minBandwidth_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
// mean performance statistics for whole run
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Mean Performance for " << totalSamples_ << " frames, duration "
         << duration_ << " seconds" << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << meanBandwidth_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Rate (Frames/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << meanRate_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Latency (us/frame)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << meanLatency_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;

  *out << "</table>" << endl;

  *out << "  </td>"                                                  << endl;
  *out << "</table>"                                                 << endl;
// now for FU sender list statistics
  *out << "<hr/>"                                                    << endl;
  *out << "<table>"                                                  << endl;
  *out << "<tr valign=\"top\">"                                      << endl;
  *out << "  <td>"                                                   << endl;

  *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">" << endl;
  *out << "<colgroup> <colgroup align=\"rigth\">"                    << endl;
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "FU Sender Information"                            << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;

    *out << "<tr>" << endl;
    *out << "<th >" << endl;
    *out << "Parameter" << endl;
    *out << "</th>" << endl;
    *out << "<th>" << endl;
    *out << "Value" << endl;
    *out << "</th>" << endl;
    *out << "</tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Number of FU Senders" << endl;
          *out << "</td>" << endl;
          *out << "<td>" << endl;
          *out << smfusenders_.size() << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;

  *out << "</table>" << endl;

  *out << "  </td>"                                                  << endl;
  *out << "</table>"                                                 << endl;
  //---- separate pages for FU senders and Streamer Output
  *out << "<hr/>"                                                 << endl;
  std::string url = getApplicationDescriptor()->getContextDescriptor()->getURL();
  std::string urn = getApplicationDescriptor()->getURN();
  *out << "<a href=\"" << url << "/" << urn << "/fusenderlist" << "\">" 
       << "FU Sender list web page" << "</a>" << endl;
  *out << "<hr/>"                                                 << endl;
  *out << "<a href=\"" << url << "/" << urn << "/streameroutput" << "\">" 
       << "Streamer Output Status web page" << "</a>" << endl;
  /* --- leave these here to debug event server problems
  *out << "<a href=\"" << url << "/" << urn << "/geteventdata" << "\">" 
       << "Get an event via a web page" << "</a>" << endl;
  *out << "<hr/>"                                                 << endl;
  *out << "<a href=\"" << url << "/" << urn << "/getregdata" << "\">" 
       << "Get a header via a web page" << "</a>" << endl;
  */

  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;
}


//////////// *** fusender web page //////////////////////////////////////////////////////////
void StorageManager::fusenderWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  *out << "<html>"                                                   << endl;
  *out << "<head>"                                                   << endl;
  *out << "<link type=\"text/css\" rel=\"stylesheet\"";
  *out << " href=\"/" <<  getApplicationDescriptor()->getURN()
       << "/styles.css\"/>"                   << endl;
  *out << "<title>" << getApplicationDescriptor()->getClassName() << " instance "
       << getApplicationDescriptor()->getInstance()
       << "</title>"     << endl;
    *out << "<table border=\"0\" width=\"100%\">"                      << endl;
    *out << "<tr>"                                                     << endl;
    *out << "  <td align=\"left\">"                                    << endl;
    *out << "    <img"                                                 << endl;
    *out << "     align=\"middle\""                                    << endl;
    *out << "     src=\"/daq/evb/examples/fu/images/fu64x64.gif\""     << endl;
    *out << "     alt=\"main\""                                        << endl;
    *out << "     width=\"64\""                                        << endl;
    *out << "     height=\"64\""                                       << endl;
    *out << "     border=\"\"/>"                                       << endl;
    *out << "    <b>"                                                  << endl;
    *out << getApplicationDescriptor()->getClassName() << " instance "
         << getApplicationDescriptor()->getInstance()                  << endl;
    *out << "      " << fsm_.stateName()->toString()                   << endl;
    *out << "    </b>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "    <a href=\"/urn:xdaq-application:lid=3\">"             << endl;
    *out << "      <img"                                               << endl;
    *out << "       align=\"middle\""                                  << endl;
    *out << "       src=\"/daq/xdaq/hyperdaq/images/HyperDAQ.jpg\""    << endl;
    *out << "       alt=\"HyperDAQ\""                                  << endl;
    *out << "       width=\"32\""                                      << endl;
    *out << "       height=\"32\""                                      << endl;
    *out << "       border=\"\"/>"                                     << endl;
    *out << "    </a>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "    <a href=\"/" << getApplicationDescriptor()->getURN()
         << "/debug\">"                   << endl;
    *out << "      <img"                                               << endl;
    *out << "       align=\"middle\""                                  << endl;
    *out << "       src=\"/daq/evb/bu/images/debug32x32.gif\""         << endl;
    *out << "       alt=\"debug\""                                     << endl;
    *out << "       width=\"32\""                                      << endl;
    *out << "       height=\"32\""                                     << endl;
    *out << "       border=\"\"/>"                                     << endl;
    *out << "    </a>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "</tr>"                                                    << endl;
    if(fsm_.stateName()->value_ == "Failed")
    {
      *out << "<tr>"					     << endl;
      *out << " <td>"					     << endl;
      *out << "<textarea rows=" << 5 << " cols=60 scroll=yes";
      *out << " readonly title=\"Reason For Failed\">"		     << endl;
      *out << reasonForFailedState_                                  << endl;
      *out << "</textarea>"                                          << endl;
      *out << " </td>"					     << endl;
      *out << "</tr>"					     << endl;
    }
    *out << "</table>"                                                 << endl;

  *out << "<hr/>"                                                    << endl;

// now for FU sender list statistics
  *out << "<table>"                                                  << endl;
  *out << "<tr valign=\"top\">"                                      << endl;
  *out << "  <td>"                                                   << endl;

  *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">" << endl;
  *out << "<colgroup> <colgroup align=\"rigth\">"                    << endl;
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "FU Sender List"                            << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;

    *out << "<tr>" << endl;
    *out << "<th >" << endl;
    *out << "Parameter" << endl;
    *out << "</th>" << endl;
    *out << "<th>" << endl;
    *out << "Value" << endl;
    *out << "</th>" << endl;
    *out << "</tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Number of FU Senders" << endl;
          *out << "</td>" << endl;
          *out << "<td>" << endl;
          *out << smfusenders_.size() << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
    std::vector<boost::shared_ptr<SMFUSenderStats> > vfustats = smfusenders_.getFUSenderStats();
    if(!vfustats.empty()) {
      for(vector<boost::shared_ptr<SMFUSenderStats> >::iterator pos = vfustats.begin();
          pos != vfustats.end(); ++pos)
      {
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "FU Sender URL" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          char hlturl[MAX_I2O_SM_URLCHARS];
          copy(&(((*pos)->hltURL_)->at(0)), 
               &(((*pos)->hltURL_)->at(0)) + ((*pos)->hltURL_)->size(),
               hlturl);
          hlturl[((*pos)->hltURL_)->size()] = '\0';
          *out << hlturl << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "FU Sender Class Name" << endl;
          *out << "</td>" << endl;
          *out << "<td>" << endl;
          char hltclass[MAX_I2O_SM_URLCHARS];
          copy(&(((*pos)->hltClassName_)->at(0)), 
               &(((*pos)->hltClassName_)->at(0)) + ((*pos)->hltClassName_)->size(),
               hltclass);
          hltclass[((*pos)->hltClassName_)->size()] = '\0';
          *out << hltclass << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "FU Sender Instance" << endl;
          *out << "</td>" << endl;
          *out << "<td>" << endl;
          *out << (*pos)->hltInstance_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "FU Sender Local ID" << endl;
          *out << "</td>" << endl;
          *out << "<td>" << endl;
          *out << (*pos)->hltLocalId_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "FU Sender Tid" << endl;
          *out << "</td>" << endl;
          *out << "<td>" << endl;
          *out << (*pos)->hltTid_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Product registry" << endl;
          *out << "</td>" << endl;
          *out << "<td>" << endl;
          if((*pos)->regAllReceived_) {
            *out << "All Received" << endl;
          } else {
            *out << "Partially received" << endl;
          }
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Product registry" << endl;
          *out << "</td>" << endl;
          *out << "<td>" << endl;
          if((*pos)->regCheckedOK_) {
            *out << "Checked OK" << endl;
          } else {
            *out << "Bad" << endl;
          }
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Connection Status" << endl;
          *out << "</td>" << endl;
          *out << "<td>" << endl;
          *out << (*pos)->connectStatus_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        if((*pos)->connectStatus_ > 1) {
          *out << "<tr>" << endl;
            *out << "<td >" << endl;
            *out << "Time since last data frame (us)" << endl;
            *out << "</td>" << endl;
            *out << "<td align=right>" << endl;
            *out << (*pos)->timeWaited_ << endl;
            *out << "</td>" << endl;
          *out << "  </tr>" << endl;
          *out << "<tr>" << endl;
            *out << "<td >" << endl;
            *out << "Run number" << endl;
            *out << "</td>" << endl;
            *out << "<td align=right>" << endl;
            *out << (*pos)->runNumber_ << endl;
            *out << "</td>" << endl;
          *out << "  </tr>" << endl;
          *out << "<tr>" << endl;
            *out << "<td >" << endl;
            *out << "Running locally" << endl;
            *out << "</td>" << endl;
            *out << "<td align=right>" << endl;
            if((*pos)->isLocal_) {
              *out << "Yes" << endl;
            } else {
              *out << "No" << endl;
            }
            *out << "</td>" << endl;
          *out << "  </tr>" << endl;
          *out << "<tr>" << endl;
            *out << "<td >" << endl;
            *out << "Frames received" << endl;
            *out << "</td>" << endl;
            *out << "<td align=right>" << endl;
            *out << (*pos)->framesReceived_ << endl;
            *out << "</td>" << endl;
          *out << "  </tr>" << endl;
          *out << "<tr>" << endl;
            *out << "<td >" << endl;
            *out << "Events received" << endl;
            *out << "</td>" << endl;
            *out << "<td align=right>" << endl;
            *out << (*pos)->eventsReceived_ << endl;
            *out << "</td>" << endl;
          *out << "  </tr>" << endl;
          *out << "<tr>" << endl;
            *out << "<td >" << endl;
            *out << "Total Bytes received" << endl;
            *out << "</td>" << endl;
            *out << "<td align=right>" << endl;
            *out << (*pos)->totalSizeReceived_ << endl;
            *out << "</td>" << endl;
          *out << "  </tr>" << endl;
          if((*pos)->eventsReceived_ > 0) {
            *out << "<tr>" << endl;
              *out << "<td >" << endl;
              *out << "Last frame latency (us)" << endl;
              *out << "</td>" << endl;
              *out << "<td align=right>" << endl;
              *out << (*pos)->lastLatency_ << endl;
              *out << "</td>" << endl;
            *out << "  </tr>" << endl;
            *out << "<tr>" << endl;
              *out << "<td >" << endl;
              *out << "Average event size (Bytes)" << endl;
              *out << "</td>" << endl;
              *out << "<td align=right>" << endl;
              *out << (*pos)->totalSizeReceived_/(*pos)->eventsReceived_ << endl;
              *out << "</td>" << endl;
              *out << "<tr>" << endl;
                *out << "<td >" << endl;
                *out << "Last Run Number" << endl;
                *out << "</td>" << endl;
                *out << "<td align=right>" << endl;
                *out << (*pos)->lastRunID_ << endl;
                *out << "</td>" << endl;
              *out << "  </tr>" << endl;
              *out << "<tr>" << endl;
                *out << "<td >" << endl;
                *out << "Last Event Number" << endl;
                *out << "</td>" << endl;
                *out << "<td align=right>" << endl;
                *out << (*pos)->lastEventID_ << endl;
                *out << "</td>" << endl;
              *out << "  </tr>" << endl;
            } // events received endif
          *out << "  </tr>" << endl;
          *out << "<tr>" << endl;
            *out << "<td >" << endl;
            *out << "Total out of order frames" << endl;
            *out << "</td>" << endl;
            *out << "<td align=right>" << endl;
            *out << (*pos)->totalOutOfOrder_ << endl;
            *out << "</td>" << endl;
          *out << "  </tr>" << endl;
          *out << "<tr>" << endl;
            *out << "<td >" << endl;
            *out << "Total Bad Events" << endl;
            *out << "</td>" << endl;
            *out << "<td align=right>" << endl;
            *out << (*pos)->totalBadEvents_ << endl;
            *out << "</td>" << endl;
          *out << "  </tr>" << endl;
        } // connect status endif
      } // Sender list loop
    } //sender size test endif

  *out << "</table>" << endl;

  *out << "  </td>"                                                  << endl;
  *out << "</table>"                                                 << endl;

  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;
}


//////////// *** streamer file output web page //////////////////////////////////////////////////////////
void StorageManager::streamerOutputWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  *out << "<html>"                                                   << endl;
  *out << "<head>"                                                   << endl;
  *out << "<link type=\"text/css\" rel=\"stylesheet\"";
  *out << " href=\"/" <<  getApplicationDescriptor()->getURN()
       << "/styles.css\"/>"                   << endl;
  *out << "<title>" << getApplicationDescriptor()->getClassName() << " instance "
       << getApplicationDescriptor()->getInstance()
       << "</title>"     << endl;
    *out << "<table border=\"0\" width=\"100%\">"                      << endl;
    *out << "<tr>"                                                     << endl;
    *out << "  <td align=\"left\">"                                    << endl;
    *out << "    <img"                                                 << endl;
    *out << "     align=\"middle\""                                    << endl;
    *out << "     src=\"/daq/evb/examples/fu/images/fu64x64.gif\""     << endl;
    *out << "     alt=\"main\""                                        << endl;
    *out << "     width=\"64\""                                        << endl;
    *out << "     height=\"64\""                                       << endl;
    *out << "     border=\"\"/>"                                       << endl;
    *out << "    <b>"                                                  << endl;
    *out << getApplicationDescriptor()->getClassName() << " instance "
         << getApplicationDescriptor()->getInstance()                  << endl;
    *out << "      " << fsm_.stateName()->toString()                   << endl;
    *out << "    </b>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "    <a href=\"/urn:xdaq-application:lid=3\">"             << endl;
    *out << "      <img"                                               << endl;
    *out << "       align=\"middle\""                                  << endl;
    *out << "       src=\"/daq/xdaq/hyperdaq/images/HyperDAQ.jpg\""    << endl;
    *out << "       alt=\"HyperDAQ\""                                  << endl;
    *out << "       width=\"32\""                                      << endl;
    *out << "       height=\"32\""                                      << endl;
    *out << "       border=\"\"/>"                                     << endl;
    *out << "    </a>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "    <a href=\"/" << getApplicationDescriptor()->getURN()
         << "/debug\">"                   << endl;
    *out << "      <img"                                               << endl;
    *out << "       align=\"middle\""                                  << endl;
    *out << "       src=\"/daq/evb/bu/images/debug32x32.gif\""         << endl;
    *out << "       alt=\"debug\""                                     << endl;
    *out << "       width=\"32\""                                      << endl;
    *out << "       height=\"32\""                                     << endl;
    *out << "       border=\"\"/>"                                     << endl;
    *out << "    </a>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "</tr>"                                                    << endl;
    if(fsm_.stateName()->value_ == "Failed")
    {
      *out << "<tr>"					     << endl;
      *out << " <td>"					     << endl;
      *out << "<textarea rows=" << 5 << " cols=60 scroll=yes";
      *out << " readonly title=\"Reason For Failed\">"		     << endl;
      *out << reasonForFailedState_                                  << endl;
      *out << "</textarea>"                                          << endl;
      *out << " </td>"					     << endl;
      *out << "</tr>"					     << endl;
    }
    *out << "</table>"                                                 << endl;

    *out << "<hr/>"                                                    << endl;

  // should first test if jc_ is valid
      if(ser_prods_size_ != 0) {
        boost::mutex::scoped_lock sl(halt_lock_);
        if(jc_.use_count() != 0) {
          std::list<std::string>& files = jc_->get_filelist();
          std::list<std::string>& currfiles = jc_->get_currfiles();

          if(files.size() > 0 )
          {
            *out << "<P>#    name                             evt        size     " << endl;
            for(list<string>::const_iterator it = files.begin();
                it != files.end(); it++)
                *out << "<P> " <<*it << endl;
          }
          for(list<string>::const_iterator it = currfiles.begin();
              it != currfiles.end(); it++)
              *out << "<P>CurrentFile = " <<*it << endl;
        }
      }

  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;
}


//////////// *** get event data web page //////////////////////////////////////////////////////////
void StorageManager::eventdataWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  // default the message length to zero
  int len=0;

  // determine the consumer ID from the event request
  // message, if it is available.
  unsigned int consumerId = 0;
  std::string lengthString = in->getenv("CONTENT_LENGTH");
  unsigned long contentLength = std::atol(lengthString.c_str());
  if (contentLength > 0) 
    {
      auto_ptr< vector<char> > bufPtr(new vector<char>(contentLength));
      in->read(&(*bufPtr)[0], contentLength);
      OtherMessageView requestMessage(&(*bufPtr)[0]);
      if (requestMessage.code() == Header::EVENT_REQUEST)
	{
	  uint8 *bodyPtr = requestMessage.msgBody();
	  consumerId = convert32(bodyPtr);
	}
    }
  
  // first test if StorageManager is in Enabled state and registry is filled
  // this must be the case for valid data to be present
  if(fsm_.stateName()->toString() == "Enabled" && ser_prods_size_ != 0)
  {
    boost::shared_ptr<EventServer> eventServer;
    if (jc_.get() != NULL)
    {
      eventServer = jc_->getEventServer();
    }
    if (eventServer.get() != NULL)
    {
      boost::shared_ptr< std::vector<char> > bufPtr =
          eventServer->getEvent(consumerId);
      if (bufPtr.get() != NULL)
      {
        EventMsgView msgView(&(*bufPtr)[0]);

        unsigned char* from = msgView.startAddress();
        unsigned int dsize = msgView.size();
        if(mybuffer_.capacity() < dsize) mybuffer_.resize(dsize);
        unsigned char* pos = (unsigned char*) &mybuffer_[0];

        copy(from,from+dsize,pos);
        len = dsize;
        FDEBUG(10) << "sending event " << msgView.event() << std::endl;
      }
    }
    
    out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
    out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
    out->write((char*) &mybuffer_[0],len);
  } // else send end of run as reponse
  else
    {
      OtherMessageBuilder othermsg(&mybuffer_[0],Header::DONE);
      len = othermsg.size();
      //std::cout << "making other message code = " << othermsg.code()
      //          << " and size = " << othermsg.size() << std::endl;
      
      out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
      out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
      out->write((char*) &mybuffer_[0],len);
    }
  
  // How to block if there is no data
  // How to signal if end, and there will be no more data?
  
}


//////////// *** get header (registry) web page ////////////////////////////////////////
void StorageManager::headerdataWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  // determine the consumer ID from the header request
  // message, if it is available.
  auto_ptr< vector<char> > httpPostData;
  unsigned int consumerId = 0;
  std::string lengthString = in->getenv("CONTENT_LENGTH");
  unsigned long contentLength = std::atol(lengthString.c_str());
  if (contentLength > 0) {
    auto_ptr< vector<char> > bufPtr(new vector<char>(contentLength));
    in->read(&(*bufPtr)[0], contentLength);
    OtherMessageView requestMessage(&(*bufPtr)[0]);
    if (requestMessage.code() == Header::HEADER_REQUEST)
    {
      uint8 *bodyPtr = requestMessage.msgBody();
      consumerId = convert32(bodyPtr);
    }

    // save the post data for use outside the "if" block scope in case it is
    // useful later (it will still get deleted at the end of the method)
    httpPostData = bufPtr;
  }

  // Need to use the saved serialzied registry
  // first test if StorageManager is in Enabled state and registry is filled
  // this must be the case for valid data to be present
  if(fsm_.stateName()->toString() == "Enabled" && ser_prods_size_ != 0)
    {
      if(ser_prods_size_ == 0)
	{ // not available yet - return zero length stream, should return MsgCode NOTREADY
	  int len = 0;
	  out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
	  out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
	  out->write((char*) &mybuffer_[0],len);
	} 
      else 
	{
	  // overlay an INIT message view on the serialized
	  // products array so that we can initialize the consumer event selection
	  InitMsgView initView(&serialized_prods_[0]);
	  if (jc_.get() != NULL)
	    {
	      boost::shared_ptr<EventServer> eventServer = jc_->getEventServer();
	      if (eventServer.get() != NULL)
		{
		  boost::shared_ptr<ConsumerPipe> consPtr =
		    eventServer->getConsumer(consumerId);
		  if (consPtr.get() != NULL)
		    {
		      consPtr->initializeSelection(initView);
		    }
		}
	    }
	  unsigned int len = ser_prods_size_;
          if(mybuffer_.capacity() < len) mybuffer_.resize(len);
	  for (unsigned int i=0; i<len; ++i) mybuffer_[i]=serialized_prods_[i];
	  
	  out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
	  out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
	  out->write((char*) &mybuffer_[0],len);
	}
    } 
  else 
    {
      // In wrong state for this message - return zero length stream, should return Msg NOTREADY
      int len = 0;
      out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
      out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
      out->write((char*) &mybuffer_[0],len);
    }
  
  
  // How to block if there is no header data
  // How to signal if not yet started, so there is no registry yet?
}


////////////////////////////// consumer registration web page ////////////////////////////
void StorageManager::consumerWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  if(fsm_.stateName()->toString() == "Enabled")
  { // what is the right place for this?

  std::string consumerName = "None provided";
  std::string consumerPriority = "normal";
  std::string consumerRequest = "<>";

  // read the consumer registration message from the http input stream
  std::string lengthString = in->getenv("CONTENT_LENGTH");
  unsigned long contentLength = std::atol(lengthString.c_str());
  if (contentLength > 0)
  {
    auto_ptr< vector<char> > bufPtr(new vector<char>(contentLength));
    in->read(&(*bufPtr)[0], contentLength);
    ConsRegRequestView requestMessage(&(*bufPtr)[0]);
    consumerName = requestMessage.getConsumerName();
    consumerPriority = requestMessage.getConsumerPriority();
    std::string reqString = requestMessage.getRequestParameterSet();
    if (reqString.size() >= 2) consumerRequest = reqString;
  }

  // create the buffer to hold the registration reply message
  const int BUFFER_SIZE = 100;
  char msgBuff[BUFFER_SIZE];

  // fetch the event server
  // (it and/or the job controller may not have been created yet)
  boost::shared_ptr<EventServer> eventServer;
  if (jc_.get() != NULL)
  {
    eventServer = jc_->getEventServer();
  }

  // if no event server, tell the consumer that we're not ready
  if (eventServer.get() == NULL)
  {
    // build the registration response into the message buffer
    ConsRegResponseBuilder respMsg(msgBuff, BUFFER_SIZE,
                                   ConsRegResponseBuilder::ES_NOT_READY, 0);
    // debug message so that compiler thinks respMsg is used
    FDEBUG(20) << "Registration response size =  " <<
      respMsg.size() << std::endl;
  }
  else
  {
    // construct a parameter set from the consumer request
    boost::shared_ptr<edm::ParameterSet>
      requestParamSet(new edm::ParameterSet(consumerRequest));

    // create the local consumer interface and add it to the event server
    boost::shared_ptr<ConsumerPipe>
      consPtr(new ConsumerPipe(consumerName, consumerPriority,
                               activeConsumerTimeout_.value_,
                               idleConsumerTimeout_.value_,
                               requestParamSet));
    eventServer->addConsumer(consPtr);
    // over-ride pushmode if not set in StorageManager
    if((consumerPriority.compare("SMProxyServer") == 0) && !pushMode_)
        consPtr->setPushMode(false);

    // build the registration response into the message buffer
    ConsRegResponseBuilder respMsg(msgBuff, BUFFER_SIZE,
                                   0, consPtr->getConsumerId());
    // debug message so that compiler thinks respMsg is used
    FDEBUG(20) << "Registration response size =  " <<
      respMsg.size() << std::endl;
  }

  // send the response
  ConsRegResponseView responseMessage(msgBuff);
  unsigned int len = responseMessage.size();
  if(mybuffer_.capacity() < len) mybuffer_.resize(len);
  for (unsigned int i=0; i<len; ++i) mybuffer_[i]=msgBuff[i];

  out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
  out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
  out->write((char*) &mybuffer_[0],len);

  } else { // is this the right thing to send?
   // In wrong state for this message - return zero length stream, should return Msg NOTREADY
   int len = 0;
   out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
   out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
   out->write((char*) &mybuffer_[0],len);
  }

}

//////////// *** get DQMevent data web page //////////////////////////////////////////////////////////
void StorageManager::DQMeventdataWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  // default the message length to zero
  int len=0;

  // determine the consumer ID from the event request
  // message, if it is available.
  unsigned int consumerId = 0;
  std::string lengthString = in->getenv("CONTENT_LENGTH");
  unsigned int contentLength = std::atol(lengthString.c_str());
  if (contentLength > 0) 
  {
    auto_ptr< vector<char> > bufPtr(new vector<char>(contentLength));
    in->read(&(*bufPtr)[0], contentLength);
    OtherMessageView requestMessage(&(*bufPtr)[0]);
    // make the change below when a tag of IOPool/Streamer can be used without FW changes
    //if (requestMessage.code() == Header::DQMEVENT_REQUEST)
    if (requestMessage.code() == Header::EVENT_REQUEST)
    {
      uint8 *bodyPtr = requestMessage.msgBody();
      consumerId = convert32(bodyPtr);
    }
  }
  
  // first test if StorageManager is in Enabled state and this is a valid request
  // there must also be DQM data available
  if(fsm_.stateName()->toString() == "Enabled" && consumerId != 0)
  {
    boost::shared_ptr<DQMEventServer> eventServer;
    if (jc_.get() != NULL)
    {
      eventServer = jc_->getDQMEventServer();
    }
    if (eventServer.get() != NULL)
    {
      boost::shared_ptr< std::vector<char> > bufPtr =
        eventServer->getDQMEvent(consumerId);
      if (bufPtr.get() != NULL)
      {
        DQMEventMsgView msgView(&(*bufPtr)[0]);

        // what if mybuffer_ is used in multiple threads? Can it happen?
        unsigned char* from = msgView.startAddress();
        unsigned int dsize = msgView.size();
        if(mybuffer_.capacity() < dsize) mybuffer_.resize(dsize);
        unsigned char* pos = (unsigned char*) &mybuffer_[0];

        copy(from,from+dsize,pos);
        len = dsize;
        FDEBUG(10) << "sending update at event " << msgView.eventNumberAtUpdate() << std::endl;
      }
    }
    
    // check if zero length is sent when there is no valid data
    // i.e. on getDQMEvent, can already send zero length if request is invalid
    out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
    out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
    out->write((char*) &mybuffer_[0],len);
  } // else send DONE as reponse (could be end of a run)
  else
  {
    // not an event request or not in enabled state, just send DONE message
    OtherMessageBuilder othermsg(&mybuffer_[0],Header::DONE);
    len = othermsg.size();
      
    out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
    out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
    out->write((char*) &mybuffer_[0],len);
  }
  
}

////////////////////////////// DQM consumer registration web page ////////////////////////////
void StorageManager::DQMconsumerWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  if(fsm_.stateName()->toString() == "Enabled")
  { // We need to be in the enabled state

    std::string consumerName = "None provided";
    std::string consumerPriority = "normal";
    std::string consumerRequest = "*";

    // read the consumer registration message from the http input stream
    std::string lengthString = in->getenv("CONTENT_LENGTH");
    unsigned int contentLength = std::atol(lengthString.c_str());
    if (contentLength > 0)
    {
      auto_ptr< vector<char> > bufPtr(new vector<char>(contentLength));
      in->read(&(*bufPtr)[0], contentLength);
      ConsRegRequestView requestMessage(&(*bufPtr)[0]);
      consumerName = requestMessage.getConsumerName();
      consumerPriority = requestMessage.getConsumerPriority();
      // for DQM consumers top folder name is stored in the "parameteSet"
      std::string reqFolder = requestMessage.getRequestParameterSet();
      if (reqFolder.size() >= 1) consumerRequest = reqFolder;
    }

    // create the buffer to hold the registration reply message
    const int BUFFER_SIZE = 100;
    char msgBuff[BUFFER_SIZE];

    // fetch the DQMevent server
    // (it and/or the job controller may not have been created yet
    //  if not in the enabled state)
    boost::shared_ptr<DQMEventServer> eventServer;
    if (jc_.get() != NULL)
    {
      eventServer = jc_->getDQMEventServer();
    }

    // if no event server, tell the consumer that we're not ready
    if (eventServer.get() == NULL)
    {
      // build the registration response into the message buffer
      ConsRegResponseBuilder respMsg(msgBuff, BUFFER_SIZE,
                                     ConsRegResponseBuilder::ES_NOT_READY, 0);
      // debug message so that compiler thinks respMsg is used
      FDEBUG(20) << "Registration response size =  " <<
        respMsg.size() << std::endl;
    }
    else
    {
      // create the local consumer interface and add it to the event server
      boost::shared_ptr<DQMConsumerPipe>
        consPtr(new DQMConsumerPipe(consumerName, consumerPriority,
                                 DQMactiveConsumerTimeout_.value_,
                                 DQMidleConsumerTimeout_.value_,
                                 consumerRequest));
      eventServer->addConsumer(consPtr);
      // over-ride pushmode if not set in StorageManager
      if((consumerPriority.compare("SMProxyServer") == 0) && !pushMode_)
          consPtr->setPushMode(false);

      // initialize it straight away (should later pass in the folder name to
      // optionally change the selection on a register?
      consPtr->initializeSelection();

      // build the registration response into the message buffer
      ConsRegResponseBuilder respMsg(msgBuff, BUFFER_SIZE,
                                     0, consPtr->getConsumerId());
      // debug message so that compiler thinks respMsg is used
      FDEBUG(20) << "Registration response size =  " <<
        respMsg.size() << std::endl;
    }

    // send the response
    ConsRegResponseView responseMessage(msgBuff);
    unsigned int len = responseMessage.size();
    if(mybuffer_.capacity() < len) mybuffer_.resize(len);
    for (unsigned int i=0; i<len; ++i) mybuffer_[i]=msgBuff[i];

    out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
    out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
    out->write((char*) &mybuffer_[0],len);

  } else { // is this the right thing to send?
   // In wrong state for this message - return zero length stream, should return Msg NOTREADY
   int len = 0;
   out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
   out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
   out->write((char*) &mybuffer_[0],len);
  }

}

//------------------------------------------------------------------------------
// Everything that has to do with the flash list goes here
// 
// - setupFlashList()                  - setup variables and initialize them
// - actionPerformed(xdata::Event &e)  - update values in flash list
//------------------------------------------------------------------------------
void StorageManager::setupFlashList()
{
  //----------------------------------------------------------------------------
  // Setup the header variables
  //----------------------------------------------------------------------------
  class_    = getApplicationDescriptor()->getClassName();
  instance_ = getApplicationDescriptor()->getInstance();
  std::string url;
  url       = getApplicationDescriptor()->getContextDescriptor()->getURL();
  url      += "/";
  url      += getApplicationDescriptor()->getURN();
  url_      = url;

  //----------------------------------------------------------------------------
  // Create/Retrieve an infospace which can be monitored
  //----------------------------------------------------------------------------
  std::ostringstream oss;
  oss << "urn:xdaq-monitorable-" << class_.value_ << "-" << instance_.value_;
  toolbox::net::URN urn = this->createQualifiedInfoSpace(oss.str());
  xdata::InfoSpace *is = xdata::getInfoSpaceFactory()->get(urn.toString());

  //----------------------------------------------------------------------------
  // Publish monitor data in monitorable info space -- Head
  //----------------------------------------------------------------------------
  is->fireItemAvailable("class",                &class_);
  is->fireItemAvailable("instance",             &instance_);
  is->fireItemAvailable("runNumber",            &runNumber_);
  is->fireItemAvailable("url",                  &url_);
  // Body
  is->fireItemAvailable("receivedFrames",       &receivedFrames_);
  is->fireItemAvailable("storedEvents",         &storedEvents_);
  is->fireItemAvailable("dqmRecords",           &dqmRecords_);
  is->fireItemAvailable("storedVolume",         &storedVolume_);
  is->fireItemAvailable("memoryUsed",           &memoryUsed_);
  is->fireItemAvailable("instantBandwidth",     &instantBandwidth_);
  is->fireItemAvailable("instantRate",          &instantRate_);
  is->fireItemAvailable("instantLatency",       &instantLatency_);
  is->fireItemAvailable("maxBandwidth",         &maxBandwidth_);
  is->fireItemAvailable("minBandwidth",         &minBandwidth_);
  is->fireItemAvailable("duration",             &duration_);
  is->fireItemAvailable("totalSamples",         &totalSamples_);
  is->fireItemAvailable("meanBandwidth",        &meanBandwidth_);
  is->fireItemAvailable("meanRate",             &meanRate_);
  is->fireItemAvailable("meanLatency",          &meanLatency_);
  is->fireItemAvailable("STparameterSet",       &offConfig_);
  is->fireItemAvailable("stateName",            fsm_.stateName());
  is->fireItemAvailable("progressMarker",       &progressMarker_);
  is->fireItemAvailable("connectedFUs",         &connectedFUs_);
  is->fireItemAvailable("pushMode2Proxy",       &pushmode2proxy_);
  is->fireItemAvailable("collateDQM",           &collateDQM_);
  is->fireItemAvailable("archiveDQM",           &archiveDQM_);
  is->fireItemAvailable("purgeTimeDQM",         &purgeTimeDQM_);
  is->fireItemAvailable("readyTimeDQM",         &readyTimeDQM_);
  is->fireItemAvailable("filePrefixDQM",        &filePrefixDQM_);
  is->fireItemAvailable("useCompressionDQM",    &useCompressionDQM_);
  is->fireItemAvailable("compressionLevelDQM",  &compressionLevelDQM_);
  is->fireItemAvailable("nLogicalDisk",         &nLogicalDisk_);
  is->fireItemAvailable("fileCatalog",          &fileCatalog_);
  is->fireItemAvailable("fileName",             &fileName_);
  is->fireItemAvailable("filePath",             &filePath_);
  is->fireItemAvailable("mailboxPath",          &mailboxPath_);
  is->fireItemAvailable("setupLabel",           &setupLabel_);
  is->fireItemAvailable("highWaterMark",        &highWaterMark_);
  is->fireItemAvailable("lumiSectionTimeOut",   &lumiSectionTimeOut_);
  is->fireItemAvailable("maxESEventRate",       &maxESEventRate_);
  is->fireItemAvailable("activeConsumerTimeout",&activeConsumerTimeout_);
  is->fireItemAvailable("idleConsumerTimeout",  &idleConsumerTimeout_);
  is->fireItemAvailable("consumerQueueSize",    &consumerQueueSize_);

  //----------------------------------------------------------------------------
  // Attach listener to myCounter_ to detect retrieval event
  //----------------------------------------------------------------------------
  is->addItemRetrieveListener("class",                this);
  is->addItemRetrieveListener("instance",             this);
  is->addItemRetrieveListener("runNumber",            this);
  is->addItemRetrieveListener("url",                  this);
  // Body
  is->addItemRetrieveListener("receivedFrames",       this);
  is->addItemRetrieveListener("storedEvents",         this);
  is->addItemRetrieveListener("dqmRecords",           this);
  is->addItemRetrieveListener("storedVolume",         this);
  is->addItemRetrieveListener("memoryUsed",           this);
  is->addItemRetrieveListener("instantBandwidth",     this);
  is->addItemRetrieveListener("instantRate",          this);
  is->addItemRetrieveListener("instantLatency",       this);
  is->addItemRetrieveListener("maxBandwidth",         this);
  is->addItemRetrieveListener("minBandwidth",         this);
  is->addItemRetrieveListener("duration",             this);
  is->addItemRetrieveListener("totalSamples",         this);
  is->addItemRetrieveListener("meanBandwidth",        this);
  is->addItemRetrieveListener("meanRate",             this);
  is->addItemRetrieveListener("meanLatency",          this);
  is->addItemRetrieveListener("STparameterSet",       this);
  is->addItemRetrieveListener("stateName",            this);
  is->addItemRetrieveListener("progressMarker",       this);
  is->addItemRetrieveListener("connectedFUs",         this);
  is->addItemRetrieveListener("pushMode2Proxy",       this);
  is->addItemRetrieveListener("collateDQM",           this);
  is->addItemRetrieveListener("archiveDQM",           this);
  is->addItemRetrieveListener("purgeTimeDQM",         this);
  is->addItemRetrieveListener("readyTimeDQM",         this);
  is->addItemRetrieveListener("filePrefixDQM",        this);
  is->addItemRetrieveListener("useCompressionDQM",    this);
  is->addItemRetrieveListener("compressionLevelDQM",  this);
  is->addItemRetrieveListener("nLogicalDisk",         this);
  is->addItemRetrieveListener("fileCatalog",          this);
  is->addItemRetrieveListener("fileName",             this);
  is->addItemRetrieveListener("filePath",             this);
  is->addItemRetrieveListener("mailboxPath",          this);
  is->addItemRetrieveListener("setupLabel",           this);
  is->addItemRetrieveListener("highWaterMark",        this);
  is->addItemRetrieveListener("lumiSectionTimeOut",   this);
  is->addItemRetrieveListener("maxESEventRate",       this);
  is->addItemRetrieveListener("activeConsumerTimeout",this);
  is->addItemRetrieveListener("idleConsumerTimeout",  this);
  is->addItemRetrieveListener("consumerQueueSize",    this);
  //----------------------------------------------------------------------------
}


void StorageManager::actionPerformed(xdata::Event& e)  
{
  if (e.type() == "ItemRetrieveEvent") {
    std::ostringstream oss;
    oss << "urn:xdaq-monitorable:" << class_.value_ << ":" << instance_.value_;
    xdata::InfoSpace *is = xdata::InfoSpace::get(oss.str());

    is->lock();
    std::string item = dynamic_cast<xdata::ItemRetrieveEvent&>(e).itemName();
    // Only update those locations which are not always up to date
    if      (item == "connectedFUs")
      connectedFUs_   = smfusenders_.size();
    else if (item == "memoryUsed")
      memoryUsed_     = pool_->getMemoryUsage().getUsed();
    else if (item == "storedVolume")
      storedVolume_   = pmeter_->totalvolumemb();
    else if (item == "progressMarker")
      progressMarker_ = ProgressMarker::instance()->status();
    is->unlock();
  } 
}



void StorageManager::parseFileEntry(std::string in, std::string &out, unsigned int &nev, unsigned int &sz)
{
  unsigned int no;
  stringstream pippo;
  pippo << in;
  pippo >> no >> out >> nev >> sz;
}



bool StorageManager::configuring(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start configuring ...");
    
    
    // Get into the Ready state from Halted state
    
    try {
      if(!edmplugin::PluginManager::isAvailable()) {
        edmplugin::PluginManager::configure(edmplugin::standard::config());
      }
    }
    catch(cms::Exception& e)
    {
      reasonForFailedState_ = e.explainSelf();
      fsm_.fireFailed(reasonForFailedState_,this);
      return false;
      //XCEPT_RAISE (toolbox::fsm::exception::Exception, e.explainSelf());
    }
    
    // give the JobController a configuration string and
    // get the registry data coming over the network (the first one)
    // Note that there is currently no run number check for the INIT
    // message, just the first one received once in Enabled state is used
    evf::ParameterSetRetriever smpset(offConfig_.value_);
    
    string my_config = smpset.getAsString();
    
    pushMode_ = (bool) pushmode2proxy_;
    smConfigString_    = my_config;
    smFileCatalog_     = fileCatalog_.toString();
    
    boost::shared_ptr<stor::Parameter> smParameter_ = stor::Configurator::instance()->getParameter();
    smParameter_ -> setCloseFileScript(closeFileScript_.toString());
    smParameter_ -> setNotifyTier0Script(notifyTier0Script_.toString());
    smParameter_ -> setInsertFileScript(insertFileScript_.toString());
    smParameter_ -> setFileCatalog(fileCatalog_.toString());
    smParameter_ -> setfileName(fileName_.toString());
    smParameter_ -> setfilePath(filePath_.toString());
    smParameter_ -> setmailboxPath(mailboxPath_.toString());
    smParameter_ -> setsetupLabel(setupLabel_.toString());
    smParameter_ -> sethighWaterMark(highWaterMark_.value_);
    smParameter_ -> setlumiSectionTimeOut(lumiSectionTimeOut_.value_);

    if (maxESEventRate_ < 0.0)
      maxESEventRate_ = 0.0;
    if (DQMmaxESEventRate_ < 0.0)
      DQMmaxESEventRate_ = 0.0;
    
    xdata::Integer cutoff(1);
    if (consumerQueueSize_ < cutoff)
      consumerQueueSize_ = cutoff;
    if (DQMconsumerQueueSize_ < cutoff)
      DQMconsumerQueueSize_ = cutoff;
    
    // the rethrows below need to be XDAQ exception types (JBK)
    try {

      jc_.reset(new stor::JobController(my_config, &deleteSMBuffer));
      
      int disks(nLogicalDisk_);
      
      jc_->setNumberOfFileSystems(disks);
      jc_->setFileCatalog(smFileCatalog_);
      jc_->setSourceId(sourceId_);

      jc_->setCollateDQM(collateDQM_);
      jc_->setArchiveDQM(archiveDQM_);
      jc_->setPurgeTimeDQM(purgeTimeDQM_);
      jc_->setReadyTimeDQM(readyTimeDQM_);
      jc_->setFilePrefixDQM(filePrefixDQM_);
      jc_->setUseCompressionDQM(useCompressionDQM_);
      jc_->setCompressionLevelDQM(compressionLevelDQM_);
      
      boost::shared_ptr<EventServer>
	eventServer(new EventServer(maxESEventRate_));
      jc_->setEventServer(eventServer);
      boost::shared_ptr<DQMEventServer>
	DQMeventServer(new DQMEventServer(DQMmaxESEventRate_));
      jc_->setDQMEventServer(DQMeventServer);
    }
    catch(cms::Exception& e)
    {
      reasonForFailedState_ = e.explainSelf();
      fsm_.fireFailed(reasonForFailedState_,this);
      //XCEPT_RAISE (toolbox::fsm::exception::Exception, e.explainSelf());
      return false;
    }
    catch(std::exception& e)
    {
      reasonForFailedState_  = e.what();
      fsm_.fireFailed(reasonForFailedState_,this);
      //XCEPT_RAISE (toolbox::fsm::exception::Exception, e.what());
      return false;
    }
    catch(...)
    {
      reasonForFailedState_  = "Unknown Exception while configuring";
      fsm_.fireFailed(reasonForFailedState_,this);
      //XCEPT_RAISE (toolbox::fsm::exception::Exception, "Unknown Exception");
      return false;
    }
    
    
    LOG4CPLUS_INFO(getApplicationLogger(),"Finished configuring!");
    
    fsm_.fireEvent("ConfigureDone",this);
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "configuring FAILED: " + (string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    return false;
  }

  return false;
}


bool StorageManager::enabling(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start enabling ...");
    
    fileList_.clear();
    eventsInFile_.clear();
    fileSize_.clear();
    storedEvents_ = 0;
    dqmRecords_   = 0;
    jc_->start();

    LOG4CPLUS_INFO(getApplicationLogger(),"Finished enabling!");
    
    fsm_.fireEvent("EnableDone",this);
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "enabling FAILED: " + (string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    return false;
  }
  catch(...)
  {
    reasonForFailedState_  = "Unknown Exception while enabling";
    fsm_.fireFailed(reasonForFailedState_,this);
    return false;
  }
  
  return false;
}


bool StorageManager::stopping(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start stopping ...");

    stopAction();

    LOG4CPLUS_INFO(getApplicationLogger(),"Finished stopping!");
    
    fsm_.fireEvent("StopDone",this);
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "stopping FAILED: " + (string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    return false;
  }
  catch(...)
  {
    reasonForFailedState_  = "Unknown Exception while stopping";
    fsm_.fireFailed(reasonForFailedState_,this);
    return false;
  }
  
  return false;
}


bool StorageManager::halting(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start halting ...");

    haltAction();
    
    LOG4CPLUS_INFO(getApplicationLogger(),"Finished halting!");
    
    fsm_.fireEvent("HaltDone",this);
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "halting FAILED: " + (string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    return false;
  }
  catch(...)
  {
    reasonForFailedState_  = "Unknown Exception while halting";
    fsm_.fireFailed(reasonForFailedState_,this);
    return false;
  }
  
  return false;
}

void StorageManager::stopAction()
{
  std::list<std::string> files = jc_->get_filelist();
  std::list<std::string> currfiles= jc_->get_currfiles();
  closedFiles_ = files.size() - currfiles.size();

  unsigned int totInFile = 0;
  for(list<string>::const_iterator it = files.begin();
      it != files.end(); it++)
    {
      string name;
      unsigned int nev;
      unsigned int size;
      parseFileEntry((*it),name,nev,size);
      fileList_.push_back(name);
      eventsInFile_.push_back(nev);
      totInFile += nev;
      fileSize_.push_back(size);
      FDEBUG(5) << name << " " << nev << " " << size << std::endl;
    }

  jc_->stop();

  jc_->join();

  // should clear the event server(s) last event/queue
  boost::shared_ptr<EventServer> eventServer;
  boost::shared_ptr<DQMEventServer> dqmeventServer;
  if (jc_.get() != NULL)
  {
    eventServer = jc_->getEventServer();
    dqmeventServer = jc_->getDQMEventServer();
  }
  if (eventServer.get() != NULL) eventServer->clearQueue();
  if (dqmeventServer.get() != NULL) dqmeventServer->clearQueue();

}

void StorageManager::haltAction()
{
  stopAction();

  // make sure serialized product registry is cleared also as its used
  // to check state readiness for web transactions
  ser_prods_size_ = 0;
  pushMode_ = false;

  {
    boost::mutex::scoped_lock sl(halt_lock_);
    jc_.reset();
  }
}

bool checkDirectoryOK(std::string path)
{
  struct stat buf;

  int retVal = stat(path.c_str(), &buf);
  if(retVal !=0 )
  {
    edm::LogError("StorageManager") << "Directory " << path
                                    << " does not exist. Error=" << errno ;
  }
}


////////////////////////////////////////////////////////////////////////////////
xoap::MessageReference StorageManager::fsmCallback(xoap::MessageReference msg)
  throw (xoap::exception::Exception)
{
  return fsm_.commandCallback(msg);
}



////////////////////////////////////////////////////////////////////////////////
void StorageManager::sendDiscardMessage(unsigned int    fuID, 
					unsigned int    hltInstance,
					unsigned int    msgType,
					string          hltClassName)
{
  /*
  std::cout << "sendDiscardMessage ... " 
	    << fuID           << "  "
	    << hltInstance    << "  "
	    << msgType        << "  "
	    << hltClassName   << std::endl;
  */
    
  set<xdaq::ApplicationDescriptor*> setOfFUs=
    getApplicationContext()->getDefaultZone()->
    getApplicationDescriptors(hltClassName.c_str());
  
  for (set<xdaq::ApplicationDescriptor*>::iterator 
	 it=setOfFUs.begin();it!=setOfFUs.end();++it)
    {
      if ((*it)->getInstance()==hltInstance)
	{
	  
	  stor::FUProxy* proxy =  new stor::FUProxy(getApplicationDescriptor(),
						    *it,
						    getApplicationContext(),
						    pool_);
	  if ( msgType == I2O_FU_DATA_DISCARD )
	    proxy -> sendDataDiscard(fuID);	
	  else if ( msgType == I2O_FU_DQM_DISCARD )
	    proxy -> sendDQMDiscard(fuID);
	  else assert("Unknown discard message type" == 0);
	  delete proxy;
	}
    }
}


////////////////////////////////////////////////////////////////////////////////
// *** Provides factory method for the instantiation of SM applications
// should probably use the MACRO? Could a XDAQ version change cause problems?
extern "C" xdaq::Application
*instantiate_StorageManager(xdaq::ApplicationStub * stub)
{
  std::cout << "Going to construct a StorageManager instance "
	    << std::endl;
  return new stor::StorageManager(stub);
}

