#ifndef IOPool_Streamer_FRDEventFileWriter_h
#define IOPool_Streamer_FRDEventFileWriter_h 

// $Id: FRDEventFileWriter.h,v 1.10 2007/09/20 20:46:55 wmtan Exp $

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

  void start() {}
  void stop() {}
 
 private:

  void initialize(std::string const& name);

  std::auto_ptr<std::ofstream> ost_;
  std::string fileName_;

};
#endif
