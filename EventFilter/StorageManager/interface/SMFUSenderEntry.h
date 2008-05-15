#ifndef _smfusenderentry_h_
#define _smfusenderentry_h_

#include <exception>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "toolbox/mem/Reference.h"
#include "toolbox/Chrono.h"

namespace stor {

  typedef std::vector<toolbox::mem::Reference*> FrameRefCollection;
  typedef std::vector<unsigned char> RegData;

struct SMFUSenderRegCollection // used to stored collection of INIT messages
{
  std::vector<std::string> outModName_;
  std::map<std::string, uint32> outModName2ModId_;
  std::map<uint32, std::string> outModId2ModName_;
  std::map<std::string, uint32> registrySizeMap_;    // size of registry in bytes once received AND copied
  std::map<std::string, bool> regAllReceivedMap_;  // All Registry fragments are received or not
  std::map<std::string, bool> regCheckedOKMap_;    // // Registry checked to be same as configuration
  std::map<std::string, FrameRefCollection> frameRefsMap_; //vector of frame reference pointers
  std::map<std::string, RegData> registryDataMap_;
  std::map<std::string, uint32> totFramesMap_;  // number of frames in this fragment
  std::map<std::string, uint32> currFramesMap_; // current frames received for registry
};

struct SMFUSenderDatCollection // used to keep track of event messages
{
  std::map<std::string, unsigned int> framesReceivedMap_;
  std::map<std::string, unsigned int> eventsReceivedMap_;
  std::map<std::string, unsigned int> lastEventIDMap_;
  std::map<std::string, unsigned int> lastFrameNumMap_;
  std::map<std::string, unsigned int> lastTotalFrameNumMap_;
  std::map<std::string, unsigned long long> totalSizeReceivedMap_;// For data only
};

struct SMFUSenderEntry  // used to store each FU sender
{
  SMFUSenderEntry(const char* hltURL,
                 const char* hltClassName,
                 const unsigned int hltLocalId,
                 const unsigned int hltInstance,
                 const unsigned int hltTid,
                 const unsigned int frameCount,
                 const unsigned int numFramesToAllocate,
                 const std::string outModName,
                 const uint32 outModId,
                 toolbox::mem::Reference *ref);
  private:
  
  char          hltURL_[MAX_I2O_SM_URLCHARS];       // FU+HLT identifiers
  char          hltClassName_[MAX_I2O_SM_URLCHARS];
  unsigned int  hltLocalId_;
  unsigned int  hltInstance_;
  unsigned int  hltTid_;
  SMFUSenderRegCollection registryCollection_;
  unsigned int  connectStatus_;   // FU+HLT connection status
  double        lastLatency_;     // Latency of last frame in microseconds
  unsigned int  runNumber_;
  bool          isLocal_;         // If detected a locally sent frame chain
  // data members below are to track the data frames from this FU
  SMFUSenderDatCollection datCollection_;
  unsigned int  framesReceived_;
  unsigned int  eventsReceived_;
  unsigned int  lastEventID_;
  unsigned int  lastRunID_;
  unsigned int  totalOutOfOrder_;
  unsigned long long  totalSizeReceived_;// For data only
  unsigned int  totalBadEvents_;   // Update meaning: include original size check?
  toolbox::Chrono chrono_;         // Keep latency for connection check
  boost::mutex entry_lock_;

  bool sameURL(const char* hltURL);
  bool sameClassName(const char* hltClassName);
  bool testCompleteFUReg(const std::string outModName);
  bool copyRegistry(const std::string outModName, toolbox::mem::Reference *head);

  public:
  
  /// returns true if frame added completes the registry
  void addReg2Entry(const unsigned int frameCount, const unsigned int numFramesToAllocate,
                 const std::string outModName, const uint32 outModId,
                 toolbox::mem::Reference *ref);
  bool addFrame(const unsigned int frameCount, const unsigned int numFrames,
                toolbox::mem::Reference *ref, const std::string outModName);
  bool update4Data(const unsigned int runNumber, const unsigned int eventNumber,
                   const unsigned int frameNum, const unsigned int totalFrames,
                   const unsigned int origdatasize, const uint32 outModId);
  void setregCheckedOK(const std::string outModName, const bool status);
  bool sameOutMod(const std::string outModName);
  bool sameOutMod(const uint32 outModId);
  void setDataStatus();
  void setrunNumber(const unsigned int run);
  void setisLocal(const bool local);
  bool regIsCopied(const std::string outModName);

  
  //double getStopWTime() const;  // more const below (nothings changes inside) didn't work due to mutex
  double getStopWTime();
  boost::shared_ptr<std::vector<char> > getvhltURL();
  boost::shared_ptr<std::vector<char> > getvhltClassName();
  unsigned int gethltLocalId() const {return hltLocalId_;}
  unsigned int gethltInstance() const {return hltInstance_;}
  unsigned int gethltTid() const {return hltTid_;}
  unsigned int getnumOutMod() const {return registryCollection_.outModName_.size();}
  SMFUSenderRegCollection getRegistryCollection() const {return registryCollection_;}
  SMFUSenderDatCollection getDatCollection() const {return datCollection_;}
  unsigned int getregistrySize(const std::string outModName);
  bool         getregAllReceived(const std::string outModName);
  unsigned int gettotFrames(const std::string outModName);
  unsigned int getcurrFrames(const std::string outModName);
  bool         getregCheckedOK(const std::string outModName);
  unsigned int getconnectStatus() const {return connectStatus_;}
  double       getlastLatency() const {return lastLatency_;}
  unsigned int getrunNumber() const {return runNumber_;}
  bool         getisLocal() const {return isLocal_;}
  unsigned int getAllframesReceived() const {return framesReceived_;}
  unsigned int getframesReceived(const std::string outModName);
  unsigned int getAlleventsReceived() const {return eventsReceived_;}
  unsigned int geteventsReceived(const std::string outModName);
  unsigned int getlastEventID() const {return lastEventID_;}
  unsigned int getlastRunID() const {return lastRunID_;}
  unsigned int getlastFrameNum(const std::string outModName);
  unsigned int getlastTotalFrameNum(const std::string outModName);
  unsigned int gettotalOutOfOrder() const {return totalOutOfOrder_;}
  unsigned long long getAlltotalSizeReceived() const {return totalSizeReceived_;}
  unsigned long long gettotalSizeReceived(const std::string outModName);
  unsigned int gettotalBadEvents() const {return totalBadEvents_;}

  bool getDataStatus();  
  char* getregistryData(const std::string outModName); // const char* here needs modifying InitMsgView ctor
  bool match(const char* hltURL, const char* hltClassName, 
                             const unsigned int hltLocalId,
                             const unsigned int hltInstance, 
                             const unsigned int hltTid);
  
};
}
#endif
