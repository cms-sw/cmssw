#ifndef IOPool_Streamer_StreamerInputIndexFile_h
#define IOPool_Streamer_StreamerInputIndexFile_h

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/IndexRecords.h"

#include "boost/shared_ptr.hpp"

#include <string>

namespace edm {

  class StreamerInputIndexFile {
  /** Class for doing Index Read Operations. */
  public:
    explicit StreamerInputIndexFile(const std::string& name);
    explicit StreamerInputIndexFile(const std::vector<std::string>& names);

    ~StreamerInputIndexFile();

    const StartIndexRecord* startMessage() const { return startMsg_.get(); }

    bool eof() {return eof_; }

    const indexRecIter begin() { return indexes_.begin(); }
    const indexRecIter end() { return indexes_.end(); }
    indexRecIter sort();

  private:

    void readStartMessage(); /** Reads in Start Message */
    int  readEventMessage(); /** Reads in next EventIndex Record */

    boost::shared_ptr<std::ifstream> ist_;

    boost::shared_ptr<StartIndexRecord> startMsg_;

    bool eof_;

    uint64 eventBufPtr_;

    std::vector<char> headerBuf_;
    std::vector<char> eventBuf_;
    uint32 eventHeaderSize_;

    std::vector<boost::shared_ptr<EventIndexRecord> > indexes_;
  };

}

#endif

