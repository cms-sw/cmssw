/**
This file contains Class definitions for the Classes that Handles
Streamer and Index file IO.

StreamerFileIO: Base Class for managing IO Handlers
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

  //typedef boost::shared_ptr<ofstream> OutPtr;
  typedef std::auto_ptr<ofstream> OutPtr;
  typedef std::auto_ptr<ifstream> InPtr;

  class StreamerFileIO {
  /** 
   Base Class for managing IO Handlers
  */
  public:
     StreamerFileIO() {};
     ~StreamerFileIO(){};

     OutPtr makeOutputFile(const string name)
     /**
     Creates an output file ptr
     */
     {
       OutPtr p (new ofstream(name.c_str(), ios_base::binary | ios_base::out));
       //OutPtr p(new ofstream(name.c_str(), ios_base::binary | ios_base::out));
       if(!(*p))
         {
           throw "cannot open output file";
         }
       return p;
     }  

     InPtr makeInputFile(const string name)
     /**
     Creates an input file ptr
     */
 
     {
       InPtr p (new ifstream(name.c_str(), ios_base::binary | ios_base::in));
       if(!(*p))
         {
           throw "cannot open input file";
         }
       return p;
     }

  };

//--------------------------------------------------------

  class StreamerOutputFile : public StreamerFileIO
  /**
  Class for doing Streamer Write operations
  */
  {
  public:
     StreamerOutputFile(const string& name);
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



      void writeEOF();

  protected:

     StreamerOutputFile(){};
     void writeStart(InitMsgBuilder& inview); 
     /**
      Performs write on InitMsgBuilder type, 
      Header is written out as Start Messsage.
     */

     void writeEventHeader(EventMsgBuilder& ineview);
      /**
      Performs write on InitMsgBuilder type, 
      Event Header is written out.
      */

      //OutPtr ost_;
      ofstream* ost_;

      uint64 current_offset_;  /** Location of current ioptr */

      uint64 first_event_offset_;
      uint64 last_event_offset_;
      uint32 events_;
      uint32 run_;
  private:

     string filename_;
  };

//--------------------------------------------------------------

  class StreamerInputFile : public StreamerFileIO
  {
  public:
    StreamerInputFile(const string& name); 
    ~StreamerInputFile();

    bool next() ; /** Moves the handler to next Event Record */
    void* currentRecord() const { return (void*) currentEvMsg_; }
    /** Points to current Record */
    void* startMessage() const { return (void*) startMsg_; } 
    /** Points to File Start Header/Message */ 

    uint32 get_hlt_bit_cnt(); /** HLT Bit Count */
    uint32 get_l1_bit_cnt(); /** L1 Bit Count */

  private:

    void readStartMessage(); 
    int  readEventMessage();
    void set_hlt_l1_sizes(); /**Sets the HLT and L1 bit sizes from Start Message */
 
    //InPtr ist_;
    ifstream* ist_;

    //std::auto_ptr<InitMsgView> startMsg_;
    //std::auto_ptr<EventMsgView> currentEvMsg_;

    InitMsgView* startMsg_;
    EventMsgView* currentEvMsg_;

    string filename_; 
    vector<char> headerBuf_; /** Buffer to store file Header */
    vector<char> eventBuf_;  /** Buffer to store Event Data */

    uint32 hlt_bit_cnt_;  /** Number of HLT Bits */
    uint32 l1_bit_cnt_;  /** Number of L1 Bits */
  };


//----------------------------------------------------

  class StreamerOutputIndexFile : public StreamerOutputFile
  /** Class for doing Index write operations. */
  {
  public:
     StreamerOutputIndexFile(const string& name);

     //Magic# and Reserved fileds 
     void writeIndexFileHeader(uint32 magicNumber, uint64 reserved);

     void write(InitMsgBuilder&);
     void write(EventMsgBuilder&, long long);

  };

//-------------------------------------------------------

  /** Struct Representing Start of File record in Index file
      MaigicNumber(04Bytes)+Reserved(08Bytes)+InitHeader 
  */

  struct StartIndexRecord {
       uint32 magic;
       uint64 reserved;
       InitMsgView* init;
       
  };

  /** Struct represents a Event filed in Streamer file.
   EventHeader + offset(64Bit)
  */

  struct EventIndexRecord {
	EventMsgView* eview;
        long long offset;
  };  

//---------------------------------------------------------

  class StreamerInputIndexFile : public StreamerFileIO
  {
  /** Class for doing Index Read Operations. */
  public:
    StreamerInputIndexFile(const string& name);
    ~StreamerInputIndexFile();

    bool next(); /**Move the ptr to next avialable record or return false*/
    void* currentRecord() const { return (void*) &currentEvMsg_; }
    void* startMessage() const { return (void*) &startMsg_; }

    uint32 get_hlt_bit_cnt(); /** HLT Bit Count */
    uint32 get_l1_bit_cnt(); /** L1 Bit Count */

  private:

    void readStartMessage(); /** Reads in Start Message */
    int  readEventMessage(); /** Reads in next EventIndex Record */
    void set_hlt_l1_sizes(); /**Sets the HLT and L1 bit sizes from Start Message */    
    string filename_; 
    //InPtr ist_;
    ifstream* ist_;
    //std::auto_ptr<InitMsgView> startMsg_;
    //std::auto_ptr<EventMsgView> currentEvMsg_;


    StartIndexRecord startMsg_;
    EventIndexRecord currentEvMsg_;

    vector<char> headerBuf_;
    vector<char> eventBuf_;
     
    uint32 hlt_bit_cnt_;  /** Number of HLT Bits */
    uint32 l1_bit_cnt_;  /** Number of L1 Bits */

  };


//-------------------------------------------

#endif


