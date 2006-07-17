#ifndef _StreamerFileReader_H
#define _StreamerFileReader_H

#include "IOPool/Streamer/interface/StreamerFileIO.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <iostream>

namespace edm
{
  class StreamerFileReader 
  {
  public:
    StreamerFileReader(edm::ParameterSet const& pset);
    ~StreamerFileReader();

     std::auto_ptr<InitMsgView> getHeader(); 
     std::auto_ptr<EventMsgView> getNextEvent();

  private:  
     string filename_;
     std::auto_ptr<StreamerInputFile> stream_reader_;
  };

} //end-of-namespace-def

#endif

