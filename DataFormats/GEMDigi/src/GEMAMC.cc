#include <cstdint>
#include "DataFormats/GEMDigi/interface/GEMAMC.h"
#include <iostream>

void GEMAMC::setAMCheader1(uint32_t dataLength, uint16_t bxID, uint32_t l1AID, uint8_t AMCnum) {
  AMCheader1 u{0};
  u.dataLength = dataLength;
  u.bxID = bxID;
  u.l1AID = l1AID;
  u.AMCnum = AMCnum;
  amch1_ = u.word;

  AMCTrailer ut{0};
  ut.dataLength = dataLength;
  ut.l1AID = l1AID;
  amct_ = ut.word;
}

void GEMAMC::setAMCheader2(uint16_t boardID, uint16_t orbitNum, uint8_t runType) {
  AMCheader2 u{0};
  u.boardID = boardID;
  u.orbitNum = orbitNum;
  u.runType = runType;
  amch2_ = u.word;
}

void GEMAMC::setGEMeventHeader(uint8_t davCnt, uint32_t davList) {
  EventHeader u{0};
  u.davCnt = davCnt;
  u.davList = davList;
  eh_ = u.word;

  EventTrailer ut{0};
  ut.BCL = 1;
  ut.DR = 1;
  ut.CL = 1;
  ut.ML = 1;
  et_ = ut.word;
}
