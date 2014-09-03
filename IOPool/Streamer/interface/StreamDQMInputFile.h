#ifndef IOPool_Streamer_StreamDQMInputFile_h
#define IOPool_Streamer_StreamDQMInputFile_h

#include "IOPool/Streamer/interface/DQMEventMessage.h"

#include <memory>

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

    const DQMEventMsgView*  currentRecord() const { return currentEvMsg_.get(); }
    /** Points to current Record */

  private:

    int readDQMEventMessage();

    std::shared_ptr<DQMEventMsgView> currentEvMsg_;
    std::auto_ptr<std::ifstream> ist_;
    std::vector<char> eventBuf_;  /** Buffer to store Event Data */
  };

#endif
