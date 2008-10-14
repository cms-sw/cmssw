#ifndef IOPool_Streamer_DumpTools_h
#define IOPool_Streamer_DumpTools_h

/** File contains simple tools to dump Init and Event 
    Messages on screen.
*/

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"

void dumpInitHeader(const InitMsgView* view);
void dumpInitView(const InitMsgView* view);
void dumpStartMsg(const InitMsgView* view);
void dumpInitVerbose(const InitMsgView* view);
void dumpInit(uint8* buf);
void printBits(unsigned char c);
void dumpEventHeader(const EventMsgView* eview);
void dumpEventView(const EventMsgView* eview);
void dumpEventIndex(const EventMsgView* eview);
void dumpEvent(uint8* buf);
void dumpDQMEventHeader(const DQMEventMsgView* dview);
void dumpDQMEventView(const DQMEventMsgView* dview);
void dumpFRDEventView(const FRDEventMsgView* fview);

#endif

