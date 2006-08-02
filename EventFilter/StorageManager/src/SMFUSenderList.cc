/*
        For saving the FU sender list

*/

#include "EventFilter/StorageManager/interface/SMFUSenderList.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

using namespace stor;
using namespace std;
using namespace edm;  // for FDEBUG macro

SMFUSenderList::SMFUSenderList(const char* hltURL,
                 const char* hltClassName,
                 const unsigned long hltLocalId,
                 const unsigned long hltInstance,
                 const unsigned long hltTid,
                 const unsigned int numFramesToAllocate,
                 const unsigned long registrySize,
                 const char* registryData):
  hltLocalId_(hltLocalId), hltInstance_(hltInstance), hltTid_(hltTid),
  registrySize_(registrySize), regAllReceived_(false),
  totFrames_(numFramesToAllocate), currFrames_(0), frameRefs_(totFrames_, 0)
{
  copy(hltURL, hltURL+MAX_I2O_SM_URLCHARS, hltURL_);
  copy(hltClassName, hltClassName+MAX_I2O_SM_URLCHARS, hltClassName_);
  // don't copy in constructor now we can have fragments
  //copy(registryData, registryData+registrySize, registryData_);
  regCheckedOK_ = false;
  /*
     Connect status
     Bit 1 = 0 disconnected (was connected before) or delete it?
           = 1 connected and received registry
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
  lastFrameNum_ = 0;
  lastTotalFrameNum_ = 0;
  totalOutOfOrder_ = 0;
  totalSizeReceived_ = 0;
  totalBadEvents_ = 0;

  FDEBUG(10) << "simpleI2OReceiver: Making a SMFUSenderList struct for "
            << hltURL_ << " class " << hltClassName_  << " instance "
            << hltInstance_ << " Tid " << hltTid_ << std::endl;
}

bool SMFUSenderList::sameURL(const char* hltURL)
{
  // should really only compare the actual length!
  //FDEBUG(9) << "sameURL: testing url " << std::endl;
  //for (int i=0; i< MAX_I2O_SM_URLCHARS; i++) {
  //  if(hltURL_[i] != hltURL[i]) {
  //    FDEBUG(9) << "sameURL: failed char test at " << i << std::endl;
  //    return false;
  //  }
  //}
  int i = 0;
  while (hltURL[i] != '\0') {
    if(hltURL_[i] != hltURL[i]) {
      FDEBUG(9) << "sameURL: failed char test at " << i << std::endl;
      return false;
    }
    i = i + 1;
  }
  //FDEBUG(9) << "sameURL: same url " << std::endl;
  return true;
}

bool SMFUSenderList::sameClassName(const char* hltClassName)
{
  // should really only compare the actual length!
  //FDEBUG(9) << "sameClassName: testing classname " << std::endl;
  //for (int i=0; i< MAX_I2O_SM_URLCHARS; i++) {
  //  if(hltClassName_[i] != hltClassName[i]) {
  //    FDEBUG(9) << "sameClassName: failed char test at " << i << std::endl;
  //    return false;
  //  }
  //}
  int i = 0;
  while (hltClassName[i] != '\0') {
    if(hltClassName_[i] != hltClassName[i]) {
      FDEBUG(9) << "sameClassName: failed char test at " << i << std::endl;
      return false;
    }
    i = i + 1;
  }
  //FDEBUG(9) << "sameClassName: same classname " << std::endl;
  return true;
}

