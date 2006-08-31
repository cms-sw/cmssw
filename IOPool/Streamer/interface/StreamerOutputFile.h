#ifndef _StreamerOutputFile_h
#define _StreamerOutputFile_h

/** StreamerOutputFile: Class for doing Streamer Write operations */

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/StreamerFileIO.h"

#include <exception>
#include <fstream>
#include <iostream>

  class StreamerOutputFile
  /**
  Class for doing Streamer Write operations
  */
  {
  public:
     explicit StreamerOutputFile(const string& name);
     /**
      CTOR, takes file path name as argument
     */
     ~StreamerOutputFile();

     void write(InitMsgBuilder&);
     /**
      Performs write on InitMsgBuilder type,
      Header + Blob, both are written out.
     */
     uint64 write(EventMsgBuilder&);
     /**
      Performs write on EventMsgBuilder type,
      Header + Blob, both are written out.
      RETURNS the Offset in Stream while at
              which Event was written.
     */

      void writeEOF(uint32 statusCode,
                    std::vector<uint32>& hltStats);

  private:
    void writeEventHeader(EventMsgBuilder& ineview);
    void writeStart(InitMsgBuilder& inview);

  private:
    OutputFile streamerfile_;

};


#endif
