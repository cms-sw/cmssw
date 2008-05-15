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

unsigned int SMFUSenderList::size()
{
  boost::mutex::scoped_lock sl(list_lock_);
  return (unsigned int)fulist_.size();
}

boost::shared_ptr<stor::SMFUSenderEntry> SMFUSenderList::findEntry(const char* hltURL, 
    const char* hltClassName, 
    const unsigned int hltLocalId,
    const unsigned int hltInstance, 
    const unsigned int hltTid)
{
   // initial empty pointer
   boost::shared_ptr<stor::SMFUSenderEntry> entryPtr;
   if(fulist_.empty()) return entryPtr;

   for(list<boost::shared_ptr<stor::SMFUSenderEntry> >::iterator pos = fulist_.begin(); 
       pos != fulist_.end(); ++pos)
   {
      if((*pos)->match(hltURL, hltClassName, hltLocalId, hltInstance, hltTid))
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
    toolbox::mem::Reference *ref, const std::string outModName, const uint32 outModId)
{
   boost::shared_ptr<stor::SMFUSenderEntry> entry_p(new SMFUSenderEntry(hltURL, hltClassName,
                     hltLocalId, hltInstance, hltTid, frameCount, numFrames, 
                     outModName, outModId, ref));
   fulist_.push_back(entry_p);
   return entry_p;
}

bool stor::SMFUSenderList::eraseEntry(const char* hltURL, const char* hltClassName, 
                  const unsigned int hltLocalId,
                  const unsigned int hltInstance, 
                  const unsigned int hltTid)
{
   for(list<boost::shared_ptr<stor::SMFUSenderEntry> >::iterator pos = fulist_.begin(); 
       pos != fulist_.end(); ++pos)
   {
      if((*pos)->match(hltURL, hltClassName, hltLocalId, hltInstance, hltTid))
      {
        fulist_.erase(pos);
        return true;
      }
   }
   return false;
}

int SMFUSenderList::registerFUSender(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid,
    const unsigned int frameCount, const unsigned int numFrames,
    toolbox::mem::Reference *ref, const std::string outModName, const uint32 outModId)
{  
  // register FU sender into the list to keep its status
  // Adds to registry data if a new output module
  boost::mutex::scoped_lock sl(list_lock_);
  boost::shared_ptr<stor::SMFUSenderEntry> foundPos = findEntry(hltURL, hltClassName, hltLocalId,
                                                                hltInstance, hltTid);
  if(foundPos != NULL)
  {
    // See if this is from the same output module or a new one
    bool sameOutMod = foundPos->sameOutMod(outModName);
    if(!sameOutMod)
    {
      FDEBUG(10) << "registerFUSender: found a new output Module " << outModName << " for URL "
                 << hltURL << " and Tid " << hltTid << std::endl;
      foundPos->addReg2Entry(frameCount, numFrames, outModName, outModId, ref);
      if(foundPos->regIsCopied(outModName)) {
        return 1;
      } else {
        return 0;
      }
    } else {
      FDEBUG(10) << "registerFUSender: found another frame " << frameCount << " for URL "
                 << hltURL << " Tid " << hltTid <<  " and output module "
                 << outModName << std::endl;
      // should really check this is not a duplicate frame
      // should check if already all frames were received (indicates reconnect maybe)
      bool regComplete = foundPos->addFrame(frameCount, numFrames, ref, outModName);
      if(regComplete) {
        return 1;
      } else {
        return 0;
      }
    }
  } else {
    FDEBUG(9) << "registerFUSender: found a different FU Sender with frame " 
              << frameCount << " for URL "
              << hltURL << " and Tid " << hltTid << std::endl;
    // register (add) this FU sender to the list
    foundPos = addEntry(hltURL, hltClassName, hltLocalId, hltInstance, hltTid, 
                        frameCount, numFrames, ref, outModName, outModId);
    // ask Jim about a better design for the return from addEntry to say reg is complete
    if(foundPos == NULL)
    {
      FDEBUG(9) << "registerFUSender: registering new FU sender at " << hltURL
                << " failed! List size is " << fulist_.size() << std::endl;
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

int SMFUSenderList::updateFUSender4data(const char* hltURL,
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
  boost::shared_ptr<stor::SMFUSenderEntry> foundPos = findEntry(hltURL, hltClassName, hltLocalId,
                                                                hltInstance, hltTid);
  if(foundPos != NULL)
  {
    // check if this is the first data frame received
    if(!foundPos->getDataStatus())
    {  // had not received data before
      foundPos->setDataStatus();
      FDEBUG(9) << "updateFUSender4data: received first data frame for URL"
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
    FDEBUG(9) << "updateFUSender4data: Cannot find output module Id "
              << outModId << " in FU Sender Entry!"
              << " With URL "
              << hltURL << " class " << hltClassName  << " instance "
              << hltInstance << " Tid " << hltTid << std::endl;
    return -1;
   }
  } else {
    // problem with this data frame from non-registered FU sender
    FDEBUG(9) << "updateFUSender4data: Cannot find FU in FU Sender list!"
              << " With URL "
              << hltURL << " class " << hltClassName  << " instance "
              << hltInstance << " Tid " << hltTid << std::endl;
    return -1;
  }
}

bool SMFUSenderList::removeFUSender(const char* hltURL,
  const char* hltClassName, const unsigned int hltLocalId,
  const unsigned int hltInstance, const unsigned int hltTid)
{
  // find this FU sender in the list and update its data statistics
  boost::mutex::scoped_lock sl(list_lock_);
  bool didErase = eraseEntry(hltURL, hltClassName, hltLocalId, hltInstance, hltTid);
  return didErase;
}

void SMFUSenderList::setRegCheckedOK(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid, const std::string outModName)
{  
  boost::mutex::scoped_lock sl(list_lock_);
  boost::shared_ptr<stor::SMFUSenderEntry> foundPos = findEntry(hltURL, hltClassName, hltLocalId,
                                                                hltInstance, hltTid);
  if(foundPos != NULL)
  {
    foundPos->setregCheckedOK(outModName, true);
  }
}

char* SMFUSenderList::getRegistryData(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid, const std::string outModName)
{  
  boost::mutex::scoped_lock sl(list_lock_);
  boost::shared_ptr<stor::SMFUSenderEntry> foundPos = findEntry(hltURL, hltClassName, hltLocalId,
                                                                hltInstance, hltTid);
  if(foundPos != NULL)
  {
    return foundPos->getregistryData(outModName);
  } else {
     return NULL;
  }
}

unsigned int SMFUSenderList::getRegistrySize(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid, const std::string outModName)
{  
  boost::mutex::scoped_lock sl(list_lock_);
  boost::shared_ptr<stor::SMFUSenderEntry> foundPos = findEntry(hltURL, hltClassName, hltLocalId,
                                                                hltInstance, hltTid);
  if(foundPos != NULL)
  {
    return foundPos->getregistrySize(outModName);
  } else {
     return 0;
  }
}

std::vector<boost::shared_ptr<SMFUSenderStats> > SMFUSenderList::getFUSenderStats()
{
  boost::mutex::scoped_lock sl(list_lock_);
  std::vector<boost::shared_ptr<SMFUSenderStats> > vstat;
  if(fulist_.size() == 0) return vstat;

  for(std::list<boost::shared_ptr<stor::SMFUSenderEntry> >::iterator pos = fulist_.begin();
          pos != fulist_.end(); ++pos)  
  {
    boost::shared_ptr<SMFUSenderStats> fustat(new SMFUSenderStats((*pos)->getvhltURL(),
                                         (*pos)->getvhltClassName(),
                                         (*pos)->gethltLocalId(),
                                         (*pos)->gethltInstance(),
                                         (*pos)->gethltTid(),
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
