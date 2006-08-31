#ifndef _StreamerInputIndexFile_h
#define _StreamerInputIndexFile_h

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/IndexRecords.h"
#include <string>

  class StreamerInputIndexFile
  {
  /** Class for doing Index Read Operations. */
  public:
    explicit StreamerInputIndexFile(const string& name);
    explicit StreamerInputIndexFile(const vector<string>& names);

    ~StreamerInputIndexFile();

    const StartIndexRecord* startMessage() const { return startMsg_; }

    bool eof() {return eof_; }

    const indexRecIter begin() { return indexes_.begin(); }
    const indexRecIter end() { return indexes_.end(); }
    indexRecIter sort();

  private:

    void readStartMessage(); /** Reads in Start Message */
    int  readEventMessage(); /** Reads in next EventIndex Record */

    ifstream* ist_;

    StartIndexRecord* startMsg_;

    bool eof_;

    uint64 eventBufPtr_;

    vector<char> headerBuf_;
    vector<char> eventBuf_;
    uint32 eventHeaderSize_;

    std::vector<EventIndexRecord*> indexes_;
  };




#endif

