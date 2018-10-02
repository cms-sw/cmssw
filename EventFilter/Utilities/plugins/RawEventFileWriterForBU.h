#ifndef EVFRAWEVENTFILEWRITERFORBU
#define EVFRAWEVENTFILEWRITERFORBU

// $Id: RawEventFileWriterForBU.h,v 1.1.2.5 2013/03/28 14:56:53 aspataru Exp $

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"

#include "EventFilter/Utilities/interface/FastMonitor.h"

#include <fstream>
#include <cstdio>
#include <sys/types.h>
#include <sys/stat.h>

#include "boost/shared_array.hpp"


class RawEventFileWriterForBU 
{
 public:

  explicit RawEventFileWriterForBU(edm::ParameterSet const& ps);
  explicit RawEventFileWriterForBU(std::string const& fileName);
  ~RawEventFileWriterForBU();

  void doOutputEvent(FRDEventMsgView const& msg);
  void doOutputEvent(boost::shared_array<unsigned char>& msg) {};
  void doOutputEventFragment(unsigned char* dataPtr,
                             unsigned long dataSize);
  //void doFlushFile();
  uint32 adler32() const { return (adlerb_ << 16) | adlera_; }

  void start(){}
  void stop();
  void initialize(std::string const& destinationDir, std::string const& name, int ls);
  void endOfLS(int ls);
  bool sharedMode() const {return false;}
  void makeRunPrefix(std::string const& destinationDir);

 private:

  bool closefd(){if(outfd_>=0){close(outfd_); outfd_=-1; return true;} else return false;}
  void finishFileWrite(int ls);
  void writeJsds(); 
  int outfd_ = -1;

  int run_ = -1;
  std::string runPrefix_;

  jsoncollector::IntJ perRunEventCount_;
  jsoncollector::IntJ perRunFileCount_;
  jsoncollector::IntJ perRunLumiCount_;
  jsoncollector::IntJ perRunLastLumi_;

  jsoncollector::IntJ perLumiEventCount_;
  jsoncollector::IntJ perLumiFileCount_;
  jsoncollector::IntJ perLumiTotalEventCount_;
  jsoncollector::IntJ perLumiLostEventCount_;
  jsoncollector::IntJ perLumiSize_;


  jsoncollector::IntJ perFileEventCount_;
  jsoncollector::IntJ perFileSize_;

  jsoncollector::FastMonitor* fileMon_ = nullptr;
  jsoncollector::FastMonitor* lumiMon_ = nullptr;
  jsoncollector::FastMonitor* runMon_ = nullptr;

  jsoncollector::DataPointDefinition rawJsonDef_;
  jsoncollector::DataPointDefinition eolJsonDef_;
  jsoncollector::DataPointDefinition eorJsonDef_;
  bool writtenJSDs_=false;

  std::unique_ptr<std::ofstream> ost_;
  std::string fileName_;
  std::string destinationDir_;

  int microSleep_;

  uint32 adlera_;
  uint32 adlerb_;

  unsigned int lumiOpen_ = 0;
  unsigned int lumiClosed_ = 0;

};
#endif
