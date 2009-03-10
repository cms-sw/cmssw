/*
        For saving the FU sender list

 $Id: SMFUSenderEntry.cc,v 1.13.2.3 2008/11/16 12:22:47 biery Exp $
*/

#include "EventFilter/StorageManager/interface/SMFUSenderEntry.h"
#include "IOPool/Streamer/interface/InitMessage.h"
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
                 const uint32 rbBufferID,
                 const uint32 regSize):
  hltLocalId_(hltLocalId), 
  hltInstance_(hltInstance), 
  hltTid_(hltTid),
  rbBufferID_(rbBufferID)
{
  copy(hltURL, hltURL+MAX_I2O_SM_URLCHARS, hltURL_);
  copy(hltClassName, hltClassName+MAX_I2O_SM_URLCHARS, hltClassName_);
  registryCollection_.outModName_.push_back(outModName);
  registryCollection_.outModName2ModId_.insert(std::make_pair(outModName, outModId));
  registryCollection_.outModId2ModName_.insert(std::make_pair(outModId, outModName));
  registryCollection_.registrySizeMap_.insert(std::make_pair(outModName, regSize));
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
                               const uint32 regSize, const std::string outModName)
{
   // should test total frames is the same, and other tests are possible
   // add a received registry fragment frame for this FU Sender
   boost::mutex::scoped_lock sl(entry_lock_);
   ++(registryCollection_.currFramesMap_[outModName]);
   return(true);
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

   // 30-Jul-2008, KAB: changed the units of the return value from
   // microseconds to milliseconds
   return ((double) chrono_.dsecs() * 1000.0);
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

bool SMFUSenderEntry::regIsCopied(const std::string outModName) //const
{
   if(registryCollection_.registrySizeMap_[outModName] > 0) return true;
   else return false;
}

bool SMFUSenderEntry::matchFirst(const char* hltURL, const char* hltClassName, 
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

bool SMFUSenderEntry::match(const char* hltURL, const char* hltClassName, 
                             const unsigned int hltLocalId,
                             const unsigned int hltInstance, 
                             const unsigned int hltTid,
                             const uint32 rbBufferID,
                             const std::string outModName) //const
{
   if(hltLocalId_ == hltLocalId && hltInstance_ == hltInstance &&
      hltTid_ == hltTid && sameURL(hltURL) && sameClassName(hltClassName) &&
      rbBufferID_ == rbBufferID && sameOutMod(outModName))
   {
      return true;
   } else {
      return false;
   }
}

bool SMFUSenderEntry::matchFirst(const char* hltURL, const char* hltClassName, 
                             const unsigned int hltLocalId,
                             const unsigned int hltInstance, 
                             const unsigned int hltTid,
                             const std::string outModName) //const
{
   if(hltLocalId_ == hltLocalId && hltInstance_ == hltInstance &&
      hltTid_ == hltTid && sameURL(hltURL) && sameClassName(hltClassName) &&
      sameOutMod(outModName))
   {
      return true;
   } else {
      return false;
   }
}

bool SMFUSenderEntry::matchFirst(const char* hltURL, const char* hltClassName, 
                             const unsigned int hltLocalId,
                             const unsigned int hltInstance, 
                             const unsigned int hltTid,
                             const uint32 outModId) //const
{
   if(hltLocalId_ == hltLocalId && hltInstance_ == hltInstance &&
      hltTid_ == hltTid && sameURL(hltURL) && sameClassName(hltClassName) &&
      sameOutMod(outModId))
   {
      return true;
   } else {
      return false;
   }
}

void SMFUSenderEntry::addReg2Entry( const unsigned int frameCount, const unsigned int numFramesToAllocate,
                                    const std::string outModName, const uint32 outModId,
                                    const uint32 regSize)
{
  registryCollection_.outModName_.push_back(outModName);
  registryCollection_.outModName2ModId_.insert(std::make_pair(outModName, outModId));
  registryCollection_.outModId2ModName_.insert(std::make_pair(outModId, outModName));
  registryCollection_.registrySizeMap_.insert(std::make_pair(outModName, regSize));
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

unsigned int SMFUSenderEntry::gettotFrames(const std::string outModName)
{
  return registryCollection_.totFramesMap_[outModName];
}

unsigned int SMFUSenderEntry::getcurrFrames(const std::string outModName)
{
  return registryCollection_.currFramesMap_[outModName];
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
