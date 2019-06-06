#include <cstdint>
#include "EventFilter/GEMRawToDigi/interface/AMCdata.h"

using namespace gem;

void AMCdata::setAMCheader1(uint32_t dataLength, uint16_t bxID, uint32_t l1AID, uint8_t AMCnum) {
  AMCheader1 u;
  u.dataLength = dataLength;
  u.bxID = bxID;
  u.l1AID = l1AID;
  u.AMCnum = AMCnum;
  amch1_ = u.word;
}

void AMCdata::setAMCheader2(uint16_t boardID, uint16_t orbitNum, uint8_t runType) {
  AMCheader2 u;
  u.boardID = boardID;
  u.orbitNum = orbitNum;
  u.runType = runType;
  amch2_ = u.word;
}

void AMCdata::setGEMeventHeader(uint8_t davCnt, uint32_t davList) {
  EventHeader u;
  u.davCnt = davCnt;
  u.davList = davList;
  eh_ = u.word;
}
