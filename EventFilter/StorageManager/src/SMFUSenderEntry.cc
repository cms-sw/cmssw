/*
        For saving the FU sender list

 $Id: SMFUSenderEntry.cc,v 1.4 2007/04/26 01:01:54 hcheung Exp $
*/

#include "EventFilter/StorageManager/interface/SMFUSenderEntry.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

using namespace stor;
using namespace std;
using namespace edm;  // for FDEBUG macro

SMFUSenderEntry::SMFUSenderEntry(const char* hltURL,
                 const char* hltClassName,
                 const unsigned int hltLocalId,
                 const unsigned int hltInstance,
                 const unsigned int hltTid,
                 const unsigned int frameCount,
                 const unsigned int numFramesToAllocate,
                 const std::string outModName,
                 const uint32 outModId,
                 toolbox::mem::Reference *ref):
  hltLocalId_(hltLocalId), 
  hltInstance_(hltInstance), 
  hltTid_(hltTid)
{
  copy(hltURL, hltURL+MAX_I2O_SM_URLCHARS, hltURL_);
  copy(hltClassName, hltClassName+MAX_I2O_SM_URLCHARS, hltClassName_);
  registryCollection_.outModName_.push_back(outModName);
  registryCollection_.outModName2ModId_.insert(std::make_pair(outModName, outModId));
  registryCollection_.outModId2ModName_.insert(std::make_pair(outModId, outModName));
  registryCollection_.registrySizeMap_.insert(std::make_pair(outModName, 0));
  registryCollection_.regAllReceivedMap_.insert(std::make_pair(outModName, false));
  registryCollection_.regCheckedOKMap_.insert(std::make_pair(outModName, false));
  FrameRefCollection frameRefs(numFramesToAllocate, 0);
  frameRefs[frameCount] = ref;
  registryCollection_.frameRefsMap_.insert(std::make_pair(outModName, frameRefs));
  RegData registryData(1000000);
  registryCollection_.registryDataMap_.insert(std::make_pair(outModName, registryData));
  registryCollection_.totFramesMap_.insert(std::make_pair(outModName, numFramesToAllocate));
  registryCollection_.currFramesMap_.insert(std::make_pair(outModName, 1));
  /*
     Connect status
     Bit 1 = 0 disconnected (was connected before)
           = 1 connected and received at least one registry frame
     Bit 2 = 0 not yet received a data frame
           = 1 received at least one data frame
  */
  connectStatus_ = 1;
  lastLatency_ = 0.0;
  runNumber_ = 0;
  isLocal_ = false;
  framesReceived_ = 1;
  eventsReceived_ = 0;
  lastEventID_ = 0;
  lastRunID_ = 0;
  totalOutOfOrder_ = 0;
  totalSizeReceived_ = 0;
  totalBadEvents_ = 0;
  // initialize the datCollection TODO
  datCollection_.framesReceivedMap_.insert(std::make_pair(outModName, 1));
  datCollection_.eventsReceivedMap_.insert(std::make_pair(outModName, 0));
  datCollection_.lastEventIDMap_.insert(std::make_pair(outModName, 0));
  datCollection_.lastFrameNumMap_.insert(std::make_pair(outModName, 0));
  datCollection_.lastTotalFrameNumMap_.insert(std::make_pair(outModName, 0));
  // why is framesReceived = 0 but size zero! Because for data only?
  datCollection_.totalSizeReceivedMap_.insert(std::make_pair(outModName, 0));
  chrono_.start(0);

  FDEBUG(9) << "SMFUSenderEntry: Making a SMFUSenderEntry struct for "
            << hltURL_ << " class " << hltClassName_  << " instance "
            << hltInstance_ << " Tid " << hltTid_ << std::endl;
  // test if this single registry frame is the only one
  testCompleteFUReg(outModName);
}

bool SMFUSenderEntry::sameURL(const char* hltURL)
{
  int i = 0;
  while (hltURL[i] != '\0') {
    if(hltURL_[i] != hltURL[i]) {
      FDEBUG(9) << "sameURL: failed char test at " << i << std::endl;
      return false;
    }
    i = i + 1;
  }
  FDEBUG(10) << "sameURL: same url " << std::endl;
  return true;
}

bool SMFUSenderEntry::sameClassName(const char* hltClassName)
{
  int i = 0;
  while (hltClassName[i] != '\0') {
    if(hltClassName_[i] != hltClassName[i]) {
      FDEBUG(9) << "sameClassName: failed char test at " << i << std::endl;
      return false;
    }
    i = i + 1;
  }
  FDEBUG(10) << "sameClassName: same classname " << std::endl;
  return true;
}

bool SMFUSenderEntry::addFrame(const unsigned int frameCount, const unsigned int numFrames,
                toolbox::mem::Reference *ref, const std::string outModName)
{
   // should test total frames is the same, and other tests are possible
   // add a received registry fragment frame for this FU Sender
   boost::mutex::scoped_lock sl(entry_lock_);
   ++(registryCollection_.currFramesMap_[outModName]);
   registryCollection_.frameRefsMap_[outModName][frameCount] = ref;
   bool copyOK = testCompleteFUReg(outModName);
   return(copyOK);
}

bool SMFUSenderEntry::update4Data(const unsigned int runNumber, const unsigned int eventNumber,
                   const unsigned int frameNum, const unsigned int totalFrames,
                   const unsigned int origdatasize, const uint32 outModId)
{
   // update statistics for a received data fragment frame for this FU sender
   boost::mutex::scoped_lock sl(entry_lock_);
   std::string outModName = registryCollection_.outModId2ModName_[outModId];
   ++framesReceived_;
   ++(datCollection_.framesReceivedMap_[outModName]);
   lastRunID_ = runNumber;
   chrono_.stop(0);
   lastLatency_ = (double) chrono_.dusecs(); //microseconds
   chrono_.start(0);
   bool problemFound = false;
   bool fullEvent = false;
   if(totalFrames == 1) 
   {
      // there is only one frame in this event assume frameNum = 1!
      ++eventsReceived_;
      ++(datCollection_.eventsReceivedMap_[outModName]);
      fullEvent = true;
      lastEventID_ = eventNumber;
      datCollection_.lastEventIDMap_[outModName] = eventNumber;
      datCollection_.lastFrameNumMap_[outModName] = frameNum;
      datCollection_.lastTotalFrameNumMap_[outModName] = totalFrames;
      totalSizeReceived_ = totalSizeReceived_ + origdatasize;
      datCollection_.totalSizeReceivedMap_[outModName] += origdatasize;
   } else {
      // flag and count if frame (event fragment) out of order
      if(datCollection_.lastEventIDMap_[outModName] == eventNumber) 
      {
        // check if in order and if last frame in a chain
        if(frameNum != datCollection_.lastFrameNumMap_[outModName] + 1) {
          ++totalOutOfOrder_;
        }
        if(totalFrames != datCollection_.lastTotalFrameNumMap_[outModName]) {
          // this is a real problem! Corrupt data frame
          problemFound = true;
        }
        // if last frame in n-of-m assume it completes an event
        // frame count starts from 1
        if(frameNum == totalFrames) { //should check totalFrames
          ++eventsReceived_;
          ++(datCollection_.eventsReceivedMap_[outModName]);
          fullEvent = true;          
          // Note only to increment total size on whole events to
          // get the correct average event size
          totalSizeReceived_ = totalSizeReceived_ + origdatasize;
          datCollection_.totalSizeReceivedMap_[outModName] += origdatasize;
        } 
        datCollection_.lastFrameNumMap_[outModName] = frameNum;
      } else { // first frame from new event
        // new event (assume run number does not change)
        lastEventID_ = eventNumber;
        datCollection_.lastEventIDMap_[outModName] = eventNumber;
        if(datCollection_.lastFrameNumMap_[outModName] != datCollection_.lastTotalFrameNumMap_[outModName] &&
           datCollection_.framesReceivedMap_[outModName] != 1) {
           // missing or frame out of order (may count multiply!)
           ++totalOutOfOrder_;
        }
        datCollection_.lastFrameNumMap_[outModName] = frameNum;
        datCollection_.lastTotalFrameNumMap_[outModName] = totalFrames;
      }
   } // totalFrames=1 or not
   if(problemFound) ++totalBadEvents_;
   return fullEvent;
}

void SMFUSenderEntry::setregCheckedOK(const std::string outModName, const bool status)
{
   boost::mutex::scoped_lock sl(entry_lock_);
   registryCollection_.regCheckedOKMap_[outModName] = status;
}

void SMFUSenderEntry::setDataStatus()
{
   boost::mutex::scoped_lock sl(entry_lock_);
   // set second bit (received at least one data frame)
   connectStatus_ = connectStatus_ | 2;
}

void SMFUSenderEntry::setrunNumber(const unsigned int run)
{
   boost::mutex::scoped_lock sl(entry_lock_);
   runNumber_ = run;
}

void SMFUSenderEntry::setisLocal(const bool local)
{
   boost::mutex::scoped_lock sl(entry_lock_);
   isLocal_ = local;
}

double SMFUSenderEntry::getStopWTime() //const
{
   boost::mutex::scoped_lock sl(entry_lock_); // should check if this is needed
   chrono_.stop(0); // note that this does not actually stop the stopwatch
   return ((double) chrono_.dusecs());
}

boost::shared_ptr<std::vector<char> > SMFUSenderEntry::getvhltURL() 
{
  int i = 0;
  while (hltURL_[i] != '\0') i = i + 1;
  boost::shared_ptr<std::vector<char> > hltURL(new vector<char>);
  hltURL->resize(i);
  copy(hltURL_, hltURL_+i, &(hltURL->at(0)) );
  return hltURL;
}

boost::shared_ptr<std::vector<char> > SMFUSenderEntry::getvhltClassName() 
{
  int i = 0;
  while (hltClassName_[i] != '\0') i = i + 1;
  boost::shared_ptr<std::vector<char> > hltClassName(new vector<char>);
  hltClassName->resize(i);
  copy(hltClassName_, hltClassName_+i, &(hltClassName->at(0)) );
  return hltClassName;
}

bool SMFUSenderEntry::getDataStatus() //const
{
   // test if second bit is set
   if((connectStatus_ & 2) > 0) return true;
   else return false;
}

char* SMFUSenderEntry::getregistryData(const std::string outModName)
{
   // this could be dangerous
   return (char*) &(registryCollection_.registryDataMap_[outModName][0]);
}

bool SMFUSenderEntry::regIsCopied(const std::string outModName) //const
{
   if(registryCollection_.registrySizeMap_[outModName] > 0) return true;
   else return false;
}

bool SMFUSenderEntry::match(const char* hltURL, const char* hltClassName, 
                             const unsigned int hltLocalId,
                             const unsigned int hltInstance, 
                             const unsigned int hltTid) //const
{
   if(hltLocalId_ == hltLocalId && hltInstance_ == hltInstance &&
      hltTid_ == hltTid && sameURL(hltURL) && sameClassName(hltClassName))
   {
      return true;
   } else {
      return false;
   }
}

bool SMFUSenderEntry::testCompleteFUReg(const std::string outModName)
{
// 
// Check that a given FU Sender has sent all frames for a registry
// If so store the serialized registry and check it and return true
// 
  if(registryCollection_.totFramesMap_[outModName] == 1)
  {
    // chain is complete as there is only one frame
    toolbox::mem::Reference *head = 0;
    head = registryCollection_.frameRefsMap_[outModName][0];
    FDEBUG(10) << "testCompleteFUReg: No chain as only one frame" << std::endl;
    // copy the whole registry for each FU sender and
    // test the registry against the one being used in Storage Manager
    registryCollection_.regAllReceivedMap_[outModName] = true;
    bool copyOK = copyRegistry(outModName, head);
    // free the complete chain buffer by freeing the head
    head->release();
    return copyOK;
  }
  else
  {
    if(registryCollection_.currFramesMap_[outModName] == registryCollection_.totFramesMap_[outModName])
    {
      FDEBUG(10) << "testCompleteFUReg: Received fragment completes a chain that has " 
                 << registryCollection_.totFramesMap_[outModName]
                 << " frames " << std::endl;
      registryCollection_.regAllReceivedMap_[outModName] = true;
      toolbox::mem::Reference *head = 0;
      toolbox::mem::Reference *tail = 0;
      if(registryCollection_.totFramesMap_[outModName] > 1)
      {
        FDEBUG(10) << "testCompleteFUReg: Remaking the chain" << std::endl;
        for(int i=0; i < (int)(registryCollection_.totFramesMap_[outModName])-1 ; ++i)
        {
          FDEBUG(10) << "testCompleteFUReg: setting next reference for frame " << i << std::endl;
          head = registryCollection_.frameRefsMap_[outModName][i];
          tail = registryCollection_.frameRefsMap_[outModName][i+1];
          head->setNextReference(tail);
        }
      }
      head = registryCollection_.frameRefsMap_[outModName][0];
      FDEBUG(10) << "testCompleteFUReg: Original chain remade" << std::endl;
      // Deal with the chain
      bool copyOK = copyRegistry(outModName, head);
      // free the complete chain buffer by freeing the head
      head->release();
      return copyOK;
    } else {  // currFrames_ is not equal to totFrames_
      // If running with local transfers, a chain of I2O frames when posted only has the
      // head frame sent. So a single frame can complete a chain for local transfers.
      // We need to test for this. Must be head frame and next pointer must exist.
      if(registryCollection_.currFramesMap_[outModName] == 1) // should check if the first is always the head?
      {
        toolbox::mem::Reference *head = 0;
        toolbox::mem::Reference *next = 0;
        // can crash here if first received frame is not first frame!?
        head = registryCollection_.frameRefsMap_[outModName][0];
        // best to check the complete chain just in case!
        unsigned int tested_frames = 1;
        next = head;
        while((next=next->getNextReference())!=0) ++tested_frames;
        FDEBUG(10) << "testCompleteFUReg: Head frame has " << tested_frames-1
          << " linked frames out of " << registryCollection_.totFramesMap_[outModName]-1 << std::endl;
        if(registryCollection_.totFramesMap_[outModName] == tested_frames)
        {
          // found a complete linked chain from the leading frame
          FDEBUG(10) << "testI2OReceiver: Leading frame contains a complete linked chain"
                     << " - must be local transfer" << std::endl;
          registryCollection_.regAllReceivedMap_[outModName] = true;
          bool copyOK = copyRegistry(outModName, head);
          head->release();
          return copyOK;
        }
      } //end of test for local transfer
    } // end of test on currFrames
  } // end of test for single frame only case
  return false;
}


bool SMFUSenderEntry::copyRegistry(const std::string outModName, toolbox::mem::Reference *head)
{
  // Copy the registry fragments and save into this FU sender entry
  FDEBUG(9) << "copyAndTestRegistry: Saving and checking the registry" << std::endl;
  I2O_MESSAGE_FRAME         *stdMsg =
    (I2O_MESSAGE_FRAME*)head->getDataLocation();
  I2O_SM_PREAMBLE_MESSAGE_FRAME *msg    =
    (I2O_SM_PREAMBLE_MESSAGE_FRAME*)stdMsg;
  // get total size and check with original size
  unsigned int origsize = msg->originalSize;
  unsigned int totalsize2check = 0;
  // TODO should change registryData_ to vector<char> and put directly there!
  //typedef vector<char> vchar;
  vector<char> tempbuffer(origsize);
  if(msg->numFrames > 1)
  {
    FDEBUG(9) << "copyAndTestRegistry: populating registry buffer from chain for "
              << msg->hltURL << " and Tid " << msg->hltTid << std::endl;
    FDEBUG(9) << "copyAndTestRegistry: getting data for frame 0" << std::endl;
    FDEBUG(9) << "copyAndTestRegistry: datasize = " << msg->dataSize << std::endl;
    int sz = msg->dataSize;
    totalsize2check = totalsize2check + sz;
    if(totalsize2check > origsize) {
      std::cerr << "copyAndTestRegistry: total registry fragment size " << sz
      << " is larger than original size " << origsize 
      << " abort copy and test" << std::endl;
      registryCollection_.regCheckedOKMap_[outModName] = false;
      registryCollection_.registrySizeMap_[outModName] = 0;
      return false;
    }
    copy(msg->dataPtr(), &msg->dataPtr()[sz], &tempbuffer[0]);
    // do not need to remake the Header for the leading frame/fragment
    // as InitMsg does not contain fragment count and total size
    int next_index = sz;
    toolbox::mem::Reference *curr = 0;
    toolbox::mem::Reference *prev = head;
    for(int i=0; i < (int)(msg->numFrames)-1 ; i++)
    {
      FDEBUG(9) << "copyAndTestRegistry: getting data for frame " << i+1 << std::endl;
      curr = prev->getNextReference(); // should test if this exists!
  
      I2O_MESSAGE_FRAME         *stdMsgcurr =
        (I2O_MESSAGE_FRAME*)curr->getDataLocation();
      I2O_SM_PREAMBLE_MESSAGE_FRAME *msgcurr    =
        (I2O_SM_PREAMBLE_MESSAGE_FRAME*)stdMsgcurr;
  
      FDEBUG(9) << "copyAndTestRegistry: datasize = " << msgcurr->dataSize << std::endl;
      int sz = msgcurr->dataSize;
      totalsize2check = totalsize2check + sz;
      if(totalsize2check > origsize) {
        std::cerr << "copyAndTestRegistry: total registry fragment size " << sz
                  << " is larger than original size " << origsize 
                  << " abort copy and test" << std::endl;
        registryCollection_.regCheckedOKMap_[outModName] = false;
        registryCollection_.registrySizeMap_[outModName] = 0;
        return false;
      }
      copy(msgcurr->dataPtr(), &msgcurr->dataPtr()[sz], &tempbuffer[next_index]);
      next_index = next_index + sz;
      prev = curr;
    }
    if(totalsize2check != origsize) {
      std::cerr << "copyAndTestRegistry: Error! Remade registry size " << totalsize2check
                << " not equal to original size " << origsize << std::endl;
      registryCollection_.regCheckedOKMap_[outModName] = false;
      registryCollection_.registrySizeMap_[outModName] = 0;
      return false;
    }
    // tempbuffer is filled with whole chain data
    registryCollection_.registrySizeMap_[outModName] = origsize; // is zero on create
    if(registryCollection_.registryDataMap_[outModName].capacity() < origsize) 
        registryCollection_.registryDataMap_[outModName].resize(origsize);
    copy(&tempbuffer[0], &tempbuffer[0]+origsize, 
        &(registryCollection_.registryDataMap_[outModName][0]));
  } else { // only one frame/fragment
    FDEBUG(9) << "copyAndTestRegistry: populating registry buffer from single frame for "
              << msg->hltURL << " and Tid " << msg->hltTid << std::endl;
    FDEBUG(9) << "copyAndTestRegistry: getting data for frame 0" << std::endl;
    FDEBUG(9) << "copyAndTestRegistry: datasize = " << msg->dataSize << std::endl;
    int sz = msg->dataSize;
    copy(msg->dataPtr(), &msg->dataPtr()[sz], &tempbuffer[0]);
    // tempbuffer is filled with all data
    registryCollection_.registrySizeMap_[outModName] = origsize; // is zero on create
    if(registryCollection_.registryDataMap_[outModName].capacity() < origsize) 
        registryCollection_.registryDataMap_[outModName].resize(origsize);
    copy(&tempbuffer[0], &tempbuffer[0]+origsize, 
        &(registryCollection_.registryDataMap_[outModName][0]));
  } // end of number of frames test

  // the test of subsequent registries must be done in StorageManager
  return true;
}

void SMFUSenderEntry::addReg2Entry( const unsigned int frameCount, const unsigned int numFramesToAllocate,
                 const std::string outModName, const uint32 outModId,
                 toolbox::mem::Reference *ref)
{
  registryCollection_.outModName_.push_back(outModName);
  registryCollection_.outModName2ModId_.insert(std::make_pair(outModName, outModId));
  registryCollection_.outModId2ModName_.insert(std::make_pair(outModId, outModName));
  registryCollection_.registrySizeMap_.insert(std::make_pair(outModName, 0));
  registryCollection_.regAllReceivedMap_.insert(std::make_pair(outModName, false));
  registryCollection_.regCheckedOKMap_.insert(std::make_pair(outModName, false));
  FrameRefCollection frameRefs(numFramesToAllocate, 0);
  frameRefs[frameCount] = ref;
  registryCollection_.frameRefsMap_.insert(std::make_pair(outModName, frameRefs));
  RegData registryData(1000000);
  registryCollection_.registryDataMap_.insert(std::make_pair(outModName, registryData));
  registryCollection_.totFramesMap_.insert(std::make_pair(outModName, numFramesToAllocate));
  registryCollection_.currFramesMap_.insert(std::make_pair(outModName, 1));
  ++framesReceived_;
  // initialize the datCollection
  datCollection_.framesReceivedMap_.insert(std::make_pair(outModName, 1));
  datCollection_.eventsReceivedMap_.insert(std::make_pair(outModName, 0));
  datCollection_.lastEventIDMap_.insert(std::make_pair(outModName, 0));
  datCollection_.lastFrameNumMap_.insert(std::make_pair(outModName, 0));
  datCollection_.lastTotalFrameNumMap_.insert(std::make_pair(outModName, 0));
  // why is framesReceived = 0 but size zero! (because not data, just registry)
  datCollection_.totalSizeReceivedMap_.insert(std::make_pair(outModName, 0));
  chrono_.stop(0);
  lastLatency_ = (double) chrono_.dusecs(); //microseconds
  chrono_.start(0);

  FDEBUG(9) << "SMFUSenderEntry: Making a SMFUSenderEntry struct for "
            << hltURL_ << " class " << hltClassName_  << " instance "
            << hltInstance_ << " Tid " << hltTid_ << std::endl;
  // test if this single registry frame is the only one
  testCompleteFUReg(outModName);
}

bool SMFUSenderEntry::sameOutMod(const std::string outModName)
{
  if(registryCollection_.outModName2ModId_.find(outModName) != registryCollection_.outModName2ModId_.end())
    return true;
  else
    return false;
}

bool SMFUSenderEntry::sameOutMod(const uint32 outModId)
{
  if(registryCollection_.outModId2ModName_.find(outModId) != registryCollection_.outModId2ModName_.end())
    return true;
  else
    return false;
}

unsigned int SMFUSenderEntry::getregistrySize(const std::string outModName)
{
  return registryCollection_.registrySizeMap_[outModName];
}

bool SMFUSenderEntry::getregAllReceived(const std::string outModName)
{
  return registryCollection_.regAllReceivedMap_[outModName];
}

unsigned int SMFUSenderEntry::gettotFrames(const std::string outModName)
{
  return registryCollection_.totFramesMap_[outModName];
}

unsigned int SMFUSenderEntry::getcurrFrames(const std::string outModName)
{
  return registryCollection_.currFramesMap_[outModName];
}

bool SMFUSenderEntry::getregCheckedOK(const std::string outModName)
{
  return registryCollection_.regCheckedOKMap_[outModName];
}

unsigned int SMFUSenderEntry::getframesReceived(const std::string outModName)
{
  return datCollection_.framesReceivedMap_[outModName];
}

unsigned int SMFUSenderEntry::geteventsReceived(const std::string outModName)
{
  return datCollection_.eventsReceivedMap_[outModName];
}

unsigned int SMFUSenderEntry::getlastFrameNum(const std::string outModName)
{
  return datCollection_.lastFrameNumMap_[outModName];
}

unsigned int SMFUSenderEntry::getlastTotalFrameNum(const std::string outModName)
{
  return datCollection_.lastTotalFrameNumMap_[outModName];
}

unsigned long long SMFUSenderEntry::gettotalSizeReceived(const std::string outModName)
{
  return datCollection_.totalSizeReceivedMap_[outModName];
}
