/**
This file contains Class definitions for the Classes that Handles
Streamer and Index file IO.

StreamerOutputFile: Class for doing Streamer Write operations
StreamerInputFile: Class for doing Streamer read operations.
StreamerOutputIndexFile: Class for doing Index write operations.
StreamerInputIndexFile: Class for doing Index Read Operations.
*/


#ifndef _StreamerFileIO_h
#define _StreamerFileIO_h

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"

#include <exception>
#include <fstream>
#include <iostream>

//-------------------------------------------------------
  
  /** Struct Representing Start of File record in Index file
      MaigicNumber(04Bytes)+Reserved(08Bytes)+InitHeader
  */
    
  struct StartIndexRecord {
       uint32* magic;
       uint64* reserved;
       InitMsgView* init;
       
  };

  /** Struct represents a Event filed in Streamer file.
   EventHeader + offset(64Bit)
  */

  struct EventIndexRecord {
        EventMsgView* eview; 
        uint64* offset;
  };  


/** Iterator for EventIndexRecord Vector */
typedef std::vector<EventIndexRecord>::iterator indexRecIter;

//---------------------------------------------------------

  //These defs might come handy later
  //typedef boost::shared_ptr<ofstream> OutPtr;
  typedef std::auto_ptr<ofstream> OutPtr;
  typedef std::auto_ptr<ifstream> InPtr;


//--------------------------------------------------------

  class OutputFile 
  /**
  Class representing Output (Streamer/Index) file.
  */
  {
  public:
     explicit OutputFile(const string& name);
     /**
      CTOR, takes file path name as argument
     */
     ~OutputFile();

      ofstream* ost() {return ost_;}
      string fileName() const {return filename_;}

      uint64 current_offset_;  /** Location of current ioptr */
      uint64 first_event_offset_;
      uint64 last_event_offset_;
      uint32 events_;
      uint32 run_;

   private:
     ofstream* ost_;
     string filename_; 
  };



//-------------------------------------------------------------

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

//--------------------------------------------------------------

class StreamerInputIndexFile;

  class StreamerInputFile
  {
  public:

    /**Reads a Streamer file */
    explicit StreamerInputFile(const string& name);  

    /** Reads a Streamer file and browse it through an index file */
    /** Index file name provided here */
    StreamerInputFile(const string& name, const string& order); 
 
    /** Index file reference is provided */
    StreamerInputFile(const string& name, const StreamerInputIndexFile& order);

    /** Multiple Index files for Single Streamer file */
    //StreamerInputFile(const vector<string>& names);

    ~StreamerInputFile();

    bool next() ; /** Moves the handler to next Event Record */
    void* currentRecord() const { return (void*) currentEvMsg_; }
    /** Points to current Record */
    void* startMessage() const { return (void*) startMsg_; } 
    /** Points to File Start Header/Message */ 

    uint32 get_hlt_bit_cnt(); /** HLT Bit Count */
    uint32 get_l1_bit_cnt(); /** L1 Bit Count */

    StreamerInputIndexFile* index(); /** Return pointer to current index */

  private:

    void readStartMessage(); 
    int  readEventMessage();
    void set_hlt_l1_sizes(); /**Sets the HLT and L1 bit sizes from Start Message */
 
    //InPtr ist_;
    ifstream* ist_;

    bool useIndex_;
    StreamerInputIndexFile* index_;
    indexRecIter indexIter_b;
    indexRecIter indexIter_e;

    uint32 hlt_bit_cnt_;  /** Number of HLT Bits */
    uint32 l1_bit_cnt_;  /** Number of L1 Bits */
    
    //std::auto_ptr<InitMsgView> startMsg_;
    //std::auto_ptr<EventMsgView> currentEvMsg_;
    InitMsgView* startMsg_;
    EventMsgView* currentEvMsg_;

    vector<char> headerBuf_; /** Buffer to store file Header */
    vector<char> eventBuf_;  /** Buffer to store Event Data */

  };


//----------------------------------------------------

  class StreamerOutputIndexFile 
  /** Class for doing Index write operations. */
  {
  public:
     explicit StreamerOutputIndexFile(const string& name);

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

//---------------------------------------------------------

  class StreamerInputIndexFile 
  {
  /** Class for doing Index Read Operations. */
  public:
    explicit StreamerInputIndexFile(const string& name);
    StreamerInputIndexFile(const vector<string>& names);
    ~StreamerInputIndexFile();

    //void* currentRecord() const { return (void*) &currentEvMsg_; }
    void* startMessage() const { return (void*) &startMsg_; }

    uint32 get_hlt_bit_cnt(); /** HLT Bit Count */
    uint32 get_l1_bit_cnt(); /** L1 Bit Count */
    
    bool eof() {return eof_; }

    std::vector<EventIndexRecord> indexes_;

    indexRecIter begin() { return indexes_.begin(); }
    indexRecIter end() { return indexes_.end(); }
   
    indexRecIter sort();

  private:

    void readStartMessage(); /** Reads in Start Message */
    int  readEventMessage(); /** Reads in next EventIndex Record */
    void set_hlt_l1_sizes(); /**Sets the HLT/L1 bit sizes from Start Message */

    ifstream* ist_;
    //InPtr ist_;

    //std::auto_ptr<InitMsgView> startMsg_;
    //std::auto_ptr<EventMsgView> currentEvMsg_;
    StartIndexRecord startMsg_;
    //EventIndexRecord currentEvMsg_;

    uint32 hlt_bit_cnt_;  /** Number of HLT Bits */
    uint32 l1_bit_cnt_;  /** Number of L1 Bits */
 
    bool eof_;

    uint64 eventBufPtr_;

    vector<char> headerBuf_;
    vector<char> eventBuf_;
  };


//-------------------------------------------

#endif

