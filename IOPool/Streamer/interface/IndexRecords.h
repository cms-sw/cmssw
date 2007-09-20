#ifndef IOPool_Streamer_IndexRecords_h
#define IOPool_Streamer_IndexRecords_h

/** Contains definitions for calsses representing Records
in an Index file, a little different from Init and Event Message Herades */

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/MsgTools.h"

//-------------------------------------------------------
/** Struct Representing Start of File record in Index file
    MaigicNumber(04Bytes)+Reserved(08Bytes)+Init Message Header
*/

// StartIndexRecord [uint32 magic][int64 reserved][InitMsgHeader]
  struct StartIndexRecordHeader {
         uint32 magic;
         uint64 reserved;

         StartIndexRecordHeader(void* inBuf){
            magic = convert32((unsigned char*)inBuf);
            reserved = convert64((unsigned char*)inBuf+sizeof(uint32));
         }
       };

  class StartIndexRecord {
      public:

       StartIndexRecord() { }

       ~StartIndexRecord() {
           if (init != NULL) delete init;
           if (indexHeader != NULL) delete indexHeader;
        }

       void makeInit(void* buf) {   //never call makeInit twice
         init = new InitMsgView(buf);
       }

       void makeHeader(void* buf) { //never call makeHeader twice
         indexHeader = new StartIndexRecordHeader(buf);
       }

       const InitMsgView* getInit() const { return init; }
       uint32 getMagic() const {return indexHeader->magic; }
       uint64 getReserved() const {return indexHeader->reserved;}


     private:
       StartIndexRecordHeader* indexHeader;
       InitMsgView* init;
  };


//-------------------------------------------------------
  /** Struct represents a Event filed in Streamer file.
   EventHeader + offset(64Bit)
  */

 class EventIndexRecord {

  public:
   EventIndexRecord(){}
   ~EventIndexRecord(){
      if (eview !=NULL) delete eview;
      if (offset != NULL) delete offset;
   }

   void makeEvent(void* buf) {  //never call makeEvent twice
        eview = new EventMsgView(buf);
   }

   void makeOffset(void* buf) {  //never call makeOffset twice
        offset = (uint64*) new long long (convert64((unsigned char*) buf));
   }

   const uint64 getOffset() const { return *offset; }
   const EventMsgView* getEventView() const {return eview;}

 private:
    EventMsgView* eview;
    uint64* offset;
 };


/** Iterator for EventIndexRecord Vector */
typedef std::vector<EventIndexRecord*>::iterator indexRecIter;

#endif
