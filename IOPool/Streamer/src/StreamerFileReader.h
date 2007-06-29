#ifndef _StreamerFileReader_H
#define _StreamerFileReader_H

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/StreamerInputFile.h"

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

     const InitMsgView* getHeader(); 
     const EventMsgView* getNextEvent();
     const bool newHeader(); 

  private:  

     std::vector<std::string> streamerNames_; /** names of Streamer files */
     std::auto_ptr<StreamerInputFile> stream_reader_;
  };

} //end-of-namespace-def

#endif

