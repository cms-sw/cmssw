 /** Class for doing Index write operations. */

#ifndef _StreamerOutputIndexFile_h
#define _StreamerOutputIndexFile_h

#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/StreamerFileIO.h"
#include "IOPool/Streamer/interface/MsgTools.h"

#include<string>
#include<vector>

  class StreamerOutputIndexFile 
  /** Class for doing Index write operations. */
  {
  public:
     explicit StreamerOutputIndexFile(const std::string& name);

     ~StreamerOutputIndexFile();

     //Magic# and Reserved fileds
     void writeIndexFileHeader(uint32 magicNumber, uint64 reserved);
     void write(InitMsgBuilder&);
     void write(EventMsgBuilder&, long long);
     void writeEOF(uint32 statusCode,
                    std::vector<uint32>& hltStats);
    
  private:
    OutputFile indexfile_;

  };
#endif

