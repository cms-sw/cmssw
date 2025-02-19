#ifndef IOPool_Streamer_FRDEventFileWriter_h
#define IOPool_Streamer_FRDEventFileWriter_h 

// $Id: FRDEventFileWriter.h,v 1.3 2010/02/18 09:19:02 mommsen Exp $

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"

#include <fstream>

class FRDEventFileWriter 
{
 public:

  explicit FRDEventFileWriter(edm::ParameterSet const& ps);
  explicit FRDEventFileWriter(std::string const& fileName);
  ~FRDEventFileWriter();

  void doOutputEvent(FRDEventMsgView const& msg);
  void doOutputEventFragment(unsigned char* dataPtr,
                             unsigned long dataSize);

  uint32 adler32() const { return (adlerb_ << 16) | adlera_; }

  void start() {}
  void stop() {}
 
 private:

  void initialize(std::string const& name);

  std::auto_ptr<std::ofstream> ost_;
  std::string fileName_;

  uint32 adlera_;
  uint32 adlerb_;

};
#endif
