#ifndef IOPool_Streamer_StreamerInputFile_h
#define IOPool_Streamer_StreamerInputFile_h

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/IndexRecords.h"
#include "Utilities/StorageFactory/interface/IOTypes.h"
#include "Utilities/StorageFactory/interface/Storage.h"

#include <boost/shared_ptr.hpp>

#include<string>
#include<vector>

class StreamerInputIndexFile;

  class StreamerInputFile
  {
  public:

    /**Reads a Streamer file */
    explicit StreamerInputFile(const std::string& name);

    /** Reads a Streamer file and browse it through an index file */
    /** Index file name provided here */
    explicit StreamerInputFile(const std::string& name, const std::string& order);

    /** Index file reference is provided */
    explicit StreamerInputFile(const std::string& name, const StreamerInputIndexFile& order);

    /** Multiple Index files for Single Streamer file */
    explicit StreamerInputFile(const std::vector<std::string>& names);

    ~StreamerInputFile();

    bool next() ; /** Moves the handler to next Event Record */

    const InitMsgView*  startMessage() const { return startMsg_; }
    /** Points to File Start Header/Message */

    const EventMsgView*  currentRecord() const { return currentEvMsg_; }
    /** Points to current Record */

    const StreamerInputIndexFile* index(); /** Return pointer to current index */

    const bool newHeader() { bool tmp=newHeader_; newHeader_=false; return tmp;}  /** Test bit if a new header is encountered */


  private:

    void openStreamerFile(const std::string& name);
    IOSize readBytes(char *buf, IOSize nBytes);

    void readStartMessage();
    int  readEventMessage();

    bool openNextFile();
    /** Compares current File header with the newly opened file header
               Returns false in case of miss match */
    bool compareHeader();

    void logFileAction(const char* msg);

    bool useIndex_;
    StreamerInputIndexFile* index_;
    indexRecIter indexIter_b;
    indexRecIter indexIter_e;

    InitMsgView* startMsg_;
    EventMsgView* currentEvMsg_;

    std::vector<char> headerBuf_; /** Buffer to store file Header */
    std::vector<char> eventBuf_;  /** Buffer to store Event Data */

    unsigned int currentFile_; /** keeps track of which file is in use at the moment*/
    std::vector<std::string> streamerNames_; /** names of Streamer files */
    bool multiStreams_;  /** True if Multiple Streams are Read */
    std::string currentFileName_;
    bool currentFileOpen_;

    uint32 currRun_;
    uint32 currProto_;

    bool newHeader_;

    boost::shared_ptr<Storage> storage_;

    bool endOfFile_;
  };



#endif
