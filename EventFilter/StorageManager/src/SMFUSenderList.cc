/*
        For saving the FU sender list

*/

#include "EventFilter/StorageManager/interface/SMFUSenderList.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

using namespace stor;
using namespace std;
using namespace edm;  // for FDEBUG macro

SMFUSenderList::SMFUSenderList()
{
  FDEBUG(10) << "SMFUSenderList: Making a SMFUSenderList" << endl;
}

void SMFUSenderList::clear()
{
  boost::mutex::scoped_lock sl(list_lock_);
  senderlist_.clear();
  numberOfRB_ = 0;
  numberOfOM_ = 0;
}

unsigned int SMFUSenderList::size()
{
  boost::mutex::scoped_lock sl(list_lock_);
  return (unsigned int)senderlist_.size();
}

unsigned int SMFUSenderList::numberOfFU()
{
  boost::mutex::scoped_lock sl(list_lock_);
  unsigned int num_RBxOMxFU = (unsigned int)senderlist_.size();
  if(numberOfRB_ != 0 && numberOfOM_ != 0)
    return (num_RBxOMxFU/(numberOfRB_ * numberOfOM_));
  else
    return 0;
  
}

boost::shared_ptr<stor::SMFUSenderEntry> SMFUSenderList::findFirstEntry(const char* hltURL, 
    const char* hltClassName, 
    const unsigned int hltLocalId,
    const unsigned int hltInstance, 
    const unsigned int hltTid)
{
   // initial empty pointer
   boost::shared_ptr<stor::SMFUSenderEntry> entryPtr;
   if(senderlist_.empty()) return entryPtr;

   for(list<boost::shared_ptr<stor::SMFUSenderEntry> >::iterator pos = senderlist_.begin(); 
       pos != senderlist_.end(); ++pos)
   {
      if((*pos)->matchFirst(hltURL, hltClassName, hltLocalId, hltInstance, hltTid))
      {
        entryPtr = (*pos);
        return entryPtr;
      }
   }
   return entryPtr;
}

boost::shared_ptr<stor::SMFUSenderEntry> SMFUSenderList::findEntry(const char* hltURL, 
    const char* hltClassName, 
    const unsigned int hltLocalId,
    const unsigned int hltInstance, 
    const unsigned int hltTid,
    const uint32 rbBufferID,
    const std::string outModName)
{
   // initial empty pointer
   boost::shared_ptr<stor::SMFUSenderEntry> entryPtr;
   if(senderlist_.empty()) return entryPtr;

   for(list<boost::shared_ptr<stor::SMFUSenderEntry> >::iterator pos = senderlist_.begin(); 
       pos != senderlist_.end(); ++pos)
   {
      if((*pos)->match(hltURL, hltClassName, hltLocalId, hltInstance, hltTid, rbBufferID, outModName))
      {
        entryPtr = (*pos);
        return entryPtr;
      }
   }
   return entryPtr;
}

boost::shared_ptr<stor::SMFUSenderEntry> SMFUSenderList::findFirstEntry(const char* hltURL, 
    const char* hltClassName, 
    const unsigned int hltLocalId,
    const unsigned int hltInstance, 
    const unsigned int hltTid,
    const std::string outModName)
{
   // initial empty pointer
   boost::shared_ptr<stor::SMFUSenderEntry> entryPtr;
   if(senderlist_.empty()) return entryPtr;

   for(list<boost::shared_ptr<stor::SMFUSenderEntry> >::iterator pos = senderlist_.begin(); 
       pos != senderlist_.end(); ++pos)
   {
      if((*pos)->matchFirst(hltURL, hltClassName, hltLocalId, hltInstance, hltTid, outModName))
      {
        entryPtr = (*pos);
        return entryPtr;
      }
   }
   return entryPtr;
}

boost::shared_ptr<stor::SMFUSenderEntry> SMFUSenderList::findFirstEntry(const char* hltURL, 
    const char* hltClassName, 
    const unsigned int hltLocalId,
    const unsigned int hltInstance, 
    const unsigned int hltTid,
    const uint32 outModId)
{
   // initial empty pointer
   boost::shared_ptr<stor::SMFUSenderEntry> entryPtr;
   if(senderlist_.empty()) return entryPtr;

   for(list<boost::shared_ptr<stor::SMFUSenderEntry> >::iterator pos = senderlist_.begin(); 
       pos != senderlist_.end(); ++pos)
   {
      if((*pos)->matchFirst(hltURL, hltClassName, hltLocalId, hltInstance, hltTid, outModId))
      {
        entryPtr = (*pos);
        return entryPtr;
      }
   }
   return entryPtr;
}

boost::shared_ptr<stor::SMFUSenderEntry> SMFUSenderList::addEntry(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid,
    const unsigned int frameCount, const unsigned int numFrames,
    const uint32 regSize, const std::string outModName, 
    const uint32 outModId, const uint32 rbBufferID)
{
   boost::shared_ptr<stor::SMFUSenderEntry> entry_p(new SMFUSenderEntry(hltURL, hltClassName,
                     hltLocalId, hltInstance, hltTid, frameCount, numFrames, 
                     outModName, outModId, rbBufferID, regSize));
   senderlist_.push_back(entry_p);
   return entry_p;
}

/*
bool stor::SMFUSenderList::eraseFirstFUEntry(const char* hltURL, const char* hltClassName, 
                  const unsigned int hltLocalId,
                  const unsigned int hltInstance, 
                  const unsigned int hltTid)
{
   boost::mutex::scoped_lock sl(list_lock_);
   for(list<boost::shared_ptr<stor::SMFUSenderEntry> >::iterator pos = senderlist_.begin(); 
       pos != senderlist_.end(); ++pos)
   {
      if((*pos)->match(hltURL, hltClassName, hltLocalId, hltInstance, hltTid))
      {
        senderlist_.erase(pos);
        return true;
      }
   }
   return false;
}
*/

int SMFUSenderList::registerDataSender(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid,
    const unsigned int frameCount, const unsigned int numFrames,
    const uint32 regSize, const std::string outModName, 
    const uint32 outModId, const uint32 rbBufferID)
{  
  // register FU sender into the list to keep its status
  // Adds to registry data if a new output module
  boost::mutex::scoped_lock sl(list_lock_);
  boost::shared_ptr<stor::SMFUSenderEntry> foundPos = findEntry(hltURL, hltClassName, hltLocalId,
                                                                hltInstance, hltTid, rbBufferID, outModName);
  if(foundPos != NULL)
  {
    // See if this is from the same output module or a new one
    bool sameOutMod = foundPos->sameOutMod(outModName);
    if(!sameOutMod)
    {
      FDEBUG(10) << "registerDataSender: found a new output Module " << outModName << " for URL "
                 << hltURL << " and Tid " << hltTid << " rb buffer ID " << rbBufferID << std::endl;
      foundPos->addReg2Entry(frameCount, numFrames, outModName, outModId, regSize);
      if(foundPos->regIsCopied(outModName)) {
        return 1;
      } else {
        return 0;
      }
    } else {
      FDEBUG(10) << "registerDataSender: found another frame " << frameCount << " for URL "
                 << hltURL << " Tid " << hltTid <<  " rb buffer ID " << rbBufferID
                 << " and output module " << outModName << std::endl;
      // should really check this is not a duplicate frame
      // should check if already all frames were received (indicates reconnect maybe)
      bool regComplete = foundPos->addFrame(frameCount, numFrames, regSize, outModName);
      if(regComplete) {
        return 1;
      } else {
        return 0;
      }
    }
  } else {
    FDEBUG(9) << "registerDataSender: found a different FU Sender with frame " 
              << frameCount << " for URL "
              << hltURL << " and Tid " << hltTid << " rb buffer ID " << rbBufferID << std::endl;
    // try to figure out numbers of RBs, FUs, and OMs
    boost::shared_ptr<stor::SMFUSenderEntry> tempPos = findFirstEntry(hltURL, hltClassName, hltLocalId,
                                                                hltInstance, hltTid);
    if(tempPos == NULL) ++numberOfRB_;
    tempPos = findFirstEntry(hltURL, hltClassName, hltLocalId, hltInstance, hltTid, outModName);
    if(tempPos == NULL) ++numberOfOM_;
    
    // register (add) this FU sender to the list
    foundPos = addEntry(hltURL, hltClassName, hltLocalId, hltInstance, hltTid, 
                        frameCount, numFrames, regSize, outModName, outModId, rbBufferID);
    // ask Jim about a better design for the return from addEntry to say reg is complete
    if(foundPos == NULL)
    {
      FDEBUG(9) << "registerDataSender: registering new FU sender at " << hltURL
                << " failed! List size is " << senderlist_.size() << std::endl;
      return -1;
    } else {
      if(foundPos->regIsCopied(outModName)) {
        return 1;
      } else {
        return 0;
      }
    }
  }
}

int SMFUSenderList::updateSender4data(const char* hltURL,
    const char* hltClassName, 
    const unsigned int hltLocalId,
    const unsigned int hltInstance, 
    const unsigned int hltTid,
    const unsigned int runNumber, 
    const unsigned int eventNumber,
    const unsigned int frameNum, 
    const unsigned int totalFrames,
    const unsigned int origdatasize, 
    const bool isLocal,
    const uint32 outModId)
{
  // find this FU sender in the list and update its data statistics
  boost::mutex::scoped_lock sl(list_lock_);
  boost::shared_ptr<stor::SMFUSenderEntry> foundPos = findFirstEntry(hltURL, hltClassName, hltLocalId,
                                                                hltInstance, hltTid, outModId);
  if(foundPos != NULL)
  {
    // check if this is the first data frame received
    if(!foundPos->getDataStatus())
    {  // had not received data before
      foundPos->setDataStatus();
      FDEBUG(9) << "updateSender4data: received first data frame for URL"
                << hltURL << " and Tid " << hltTid << std::endl;
      foundPos->setrunNumber(runNumber);
      foundPos->setisLocal(isLocal);
    }
    // check that we have received from this output module also
   if(foundPos->sameOutMod(outModId)) {
    bool gotEvt = foundPos->update4Data(runNumber, eventNumber, frameNum,
                                          totalFrames, origdatasize, outModId);
    if(gotEvt) {
      return 1;
    } else {
      return 0;
    }
   } else {
    // problem with this data frame from non-registered output module
    FDEBUG(9) << "updateSender4data: Cannot find output module Id "
              << outModId << " in FU Sender Entry!"
              << " With URL "
              << hltURL << " class " << hltClassName  << " instance "
              << hltInstance << " Tid " << hltTid << std::endl;
    return -1;
   }
  } else {
    // problem with this data frame from non-registered FU sender
    FDEBUG(9) << "updateSender4data: Cannot find FU in FU Sender list!"
              << " With URL "
              << hltURL << " class " << hltClassName  << " instance "
              << hltInstance << " Tid " << hltTid << std::endl;
    return -1;
  }
}

/*
bool SMFUSenderList::removeDataSender(const char* hltURL,
  const char* hltClassName, const unsigned int hltLocalId,
  const unsigned int hltInstance, const unsigned int hltTid)
{
  // find this FU sender in the list and update its data statistics
  //boost::mutex::scoped_lock sl(list_lock_);  # put lock in erase method instead
  bool didErase = eraseFirstFUEntry(hltURL, hltClassName, hltLocalId, hltInstance, hltTid);
  return didErase;
}
*/

unsigned int SMFUSenderList::getRegistrySize(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid, 
    const std::string outModName, const uint32 rbBufferID)
{  
  boost::mutex::scoped_lock sl(list_lock_);
  boost::shared_ptr<stor::SMFUSenderEntry> foundPos = findEntry(hltURL, hltClassName, hltLocalId,
                                                                hltInstance, hltTid, rbBufferID, outModName);
  if(foundPos != NULL)
  {
    return foundPos->getregistrySize(outModName);
  } else {
     return 0;
  }
}

std::vector<boost::shared_ptr<SMFUSenderStats> > SMFUSenderList::getSenderStats()
{
  boost::mutex::scoped_lock sl(list_lock_);
  std::vector<boost::shared_ptr<SMFUSenderStats> > vstat;
  if(senderlist_.size() == 0) return vstat;

  for(std::list<boost::shared_ptr<stor::SMFUSenderEntry> >::iterator pos = senderlist_.begin();
          pos != senderlist_.end(); ++pos)  
  {
    boost::shared_ptr<SMFUSenderStats> fustat(new SMFUSenderStats((*pos)->getvhltURL(),
                                         (*pos)->getvhltClassName(),
                                         (*pos)->gethltLocalId(),
                                         (*pos)->gethltInstance(),
                                         (*pos)->gethltTid(),
                                         (*pos)->getrbBufferID(),
                                         (*pos)->getRegistryCollection(),
                                         (*pos)->getDatCollection(),
                                         (*pos)->getconnectStatus(),
                                         (*pos)->getlastLatency(),
                                         (*pos)->getrunNumber(),
                                         (*pos)->getisLocal(),
                                         (*pos)->getAllframesReceived(),
                                         (*pos)->getAlleventsReceived(),
                                         (*pos)->getlastEventID(),
                                         (*pos)->getlastRunID(),
                                         (*pos)->gettotalOutOfOrder(),
                                         (*pos)->getAlltotalSizeReceived(),
                                         (*pos)->gettotalBadEvents(),
                                         (*pos)->getStopWTime()));
    vstat.push_back(fustat);
  }
  return vstat;
}
