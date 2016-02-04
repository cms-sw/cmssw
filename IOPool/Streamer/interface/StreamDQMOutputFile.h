#ifndef IOPool_Streamer_StreamDQMOutputFile_h
#define IOPool_Streamer_StreamDQMOutputFile_h

/** StreamDQMOutputFile: Class for doing Streamer Write operations */

#include "IOPool/Streamer/interface/MsgTools.h"

#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "IOPool/Streamer/interface/DQMEventMsgBuilder.h"

#include "IOPool/Streamer/interface/StreamerFileIO.h"

#include "boost/shared_ptr.hpp"

#include <exception>
#include <fstream>
#include <iostream>

class StreamDQMOutputFile
  /**
  Class for doing Streamer Write operations
  */
  {
  public:
     explicit StreamDQMOutputFile(const std::string& name);
     /**
      CTOR, takes file path name as argument
     */
     ~StreamDQMOutputFile();

     /**
      Performs write on InitMsgBuilder type,
      Header + Blob, both are written out.
     */

     uint64 write(const DQMEventMsgView&);
     uint64 write(const DQMEventMsgBuilder&);

      //Returns how many bytes were written out
      //uint32 writeEOF(uint32 statusCode,
                    //const std::vector<uint32>& hltStats);

  private:
     void writeDQMEventHeader(const DQMEventMsgView& inview);
     void writeDQMEventHeader(const DQMEventMsgBuilder& inview);
     boost::shared_ptr<OutputFile> dqmstreamfile_;
};

#endif
