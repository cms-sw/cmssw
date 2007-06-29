#ifndef _StreamDQMInputFile_h
#define _StreamDQMInputFile_h

#include "IOPool/Streamer/interface/DQMEventMessage.h"

#include<string>
#include<vector>
#include<fstream>

  class StreamDQMInputFile
  {
  public:

    /**Reads a Streamer file */
    explicit StreamDQMInputFile(const std::string& name);

    ~StreamDQMInputFile();

    bool next() ; /** Moves the handler to next Event Record */

    const DQMEventMsgView*  currentRecord() const { return currentEvMsg_; }
    /** Points to current Record */

  private:

    int readDQMEventMessage();

    DQMEventMsgView* currentEvMsg_;
    std::auto_ptr<std::ifstream> ist_;
    std::vector<char> eventBuf_;  /** Buffer to store Event Data */
  };

#endif
