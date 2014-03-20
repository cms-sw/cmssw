#ifndef EVFRAWEVENTFILEWRITERFORBU
#define EVFRAWEVENTFILEWRITERFORBU

// $Id: RawEventFileWriterForBU.h,v 1.1.2.5 2013/03/28 14:56:53 aspataru Exp $

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"

#include "EventFilter/Utilities/interface/FastMonitor.h"

#include <fstream>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "boost/shared_array.hpp"

using namespace jsoncollector;

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
  void doFlushFile();
  uint32 adler32() const { return (adlerb_ << 16) | adlera_; }

  void start() {}
  void stop() {}
  void initialize(std::string const& destinationDir, std::string const& name, int ls);
  void endOfLS(int ls);
  bool sharedMode() const {return false;}
  void makeRunPrefix(std::string const& destinationDir);

  void handler(int s);
  static void staticHandler(int s) { instance->handler(s); }

 private:

  int run_ = -1;
  std::string runPrefix_;

  IntJ perLumiEventCount_;
  FastMonitor* lumiMon_;
  IntJ perFileEventCount_;
  FastMonitor* perFileMon_;

  std::auto_ptr<std::ofstream> ost_;
  int outfd_;
  std::string fileName_;
  std::string destinationDir_;

  std::string jsonDefLocation_;
  int microSleep_;

  uint32 adlera_;
  uint32 adlerb_;

  static RawEventFileWriterForBU* instance;

};
#endif
