#ifndef IOPool_Streamer_StreamerInputFile_h
#define IOPool_Streamer_StreamerInputFile_h

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "Utilities/StorageFactory/interface/IOTypes.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <memory>

#include <string>
#include <vector>

namespace edm {
  class EventSkipperByID;
  class FileCatalogItem;
  class StreamerInputFile {
  public:
    /**Reads a Streamer file */
    explicit StreamerInputFile(std::string const& name,
                               std::string const& LFN,
                               std::shared_ptr<EventSkipperByID> eventSkipperByID = std::shared_ptr<EventSkipperByID>(),
                               unsigned int prefetchMBytes = 0);
    explicit StreamerInputFile(std::string const& name,
                               std::shared_ptr<EventSkipperByID> eventSkipperByID = std::shared_ptr<EventSkipperByID>(),
                               unsigned int prefetchMBytes = 0);

    /** Multiple Streamer files */
    explicit StreamerInputFile(std::vector<FileCatalogItem> const& names,
                               std::shared_ptr<EventSkipperByID> eventSkipperByID = std::shared_ptr<EventSkipperByID>(),
                               unsigned int prefetchMBytes = 0);

    ~StreamerInputFile();

    enum class Next { kEvent, kFile, kStop };
    Next next(); /** Moves the handler to next Event Record */

    InitMsgView const* startMessage() const { return startMsg_.get(); }
    /** Points to File Start Header/Message */

    EventMsgView const* currentRecord() const { return currentEvMsg_.get(); }
    /** Points to current Record */

    bool newHeader() {
      bool tmp = newHeader_;
      newHeader_ = false;
      return tmp;
    } /** Test bit if a new header is encountered */

    void closeStreamerFile();
    bool openNextFile();

  private:
    void openStreamerFile(std::string const& name, std::string const& LFN);
    std::pair<storage::IOSize, char*> readBytes(char* buf,
                                                storage::IOSize nBytes,
                                                bool zeroCopy,
                                                unsigned int skippedHdr = 0);
    storage::IOOffset skipBytes(storage::IOSize nBytes);

    void readStartMessage();
    int readEventMessage();

    /** Compares current File header with the newly opened file header
               Returns false in case of mismatch */
    bool compareHeader();

    void logFileAction(char const* msg);

    edm::propagate_const<std::shared_ptr<InitMsgView>> startMsg_;
    edm::propagate_const<std::shared_ptr<EventMsgView>> currentEvMsg_;

    std::vector<char> headerBuf_; /** Buffer to store file Header */
    std::vector<char> eventBuf_;  /** Buffer to store Event Data */

    std::vector<char> tempBuf_; /** Buffer to store prefetched bytes */
    unsigned int tempLen_ = 0;
    unsigned int tempPos_ = 0;

    unsigned int currentFile_;                   /** keeps track of which file is in use at the moment*/
    std::vector<FileCatalogItem> streamerNames_; /** names of Streamer files */
    bool multiStreams_;                          /** True if Multiple Streams are Read */
    std::string currentFileName_;
    bool currentFileOpen_;

    edm::propagate_const<std::shared_ptr<EventSkipperByID>> eventSkipperByID_;

    uint32 currRun_;
    uint32 currProto_;

    bool newHeader_;

    edm::propagate_const<std::unique_ptr<edm::storage::Storage>> storage_;

    bool endOfFile_;
  };
}  // namespace edm

#endif
