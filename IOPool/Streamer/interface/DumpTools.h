/** File contains simple tools to dump Init and Event 
    Messages on screen.
*/

#ifndef _dump_tool_
#define _dump_tool_

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"

void dumpInitHeader(const InitMsgView* view);
void dumpInitView(const InitMsgView* view);
void dumpStartMsg(const InitMsgView* view);
void dumpInit(uint8* buf);
void printBits(unsigned char c);
void dumpEventHeader(const EventMsgView* eview);
void dumpEventView(const EventMsgView* eview);
void dumpEventIndex(const EventMsgView* eview);
void dumpEvent(uint8* buf);

#endif

