#include "L1Trigger/RPCTrigger/interface/RPCBasicTrigConfig.h"
#include "L1Trigger/RPCTrigger/interface/RPCException.h"

/// Ctor
RPCBasicTrigConfig::RPCBasicTrigConfig(RPCPacManager<RPCPacData>* pacManager) {
  m_PacManager  = pacManager;
}

/// Ctor
RPCBasicTrigConfig::RPCBasicTrigConfig() {
  m_PacManager  = nullptr;
}

/** Converts TC GB-Sorter output m_tower address <0...31> ("m_tower number continous")
* to m_tower number 2'complement*/
int RPCBasicTrigConfig::towNum2TowNum2Comp(int towNum) {
  if(towNum >= 0)
    return towNum;
  else
    return 0x3F + towNum + 1;  
}

//#############################################################################################
//
//  Simple getters and setters
//
//#############################################################################################
/**
 *
 *  returns count of Trigger Crates in system.
 *
*/
int RPCBasicTrigConfig::getTCsCnt() { return m_TRIGGER_CRATES_CNT; }

/**
 *
 * returns number og Trigger Boards in one Trigger Crate.
 *
*/
int RPCBasicTrigConfig::getTBsInTC() { return m_TB_IN_TC_CNT; }

/**
 *
 * Returns the index of TC that should run given LogCone.
 *
 *
*/
int RPCBasicTrigConfig::getTCNum(const RPCConst::l1RpcConeCrdnts& coneCrdnts) {
  return coneCrdnts.m_LogSector;
}
/**
 *
 * Returns the count of Towers (3 or 4), that are covered by given TB.
 *
*/
int RPCBasicTrigConfig::getTowsCntOnTB(int tbNum) {
  return m_TOWERS_CNT_ON_TB[tbNum];
}
/** Converts TC GB-Sorter input m_tower address <0...35> ("m_tower number natural")
 * to m_tower number <-16...0...16>
 * TC GB-Sorter input m_tower address is 8 bits: [7...2] TB num, [1...0] m_tower num on TB.*/
int RPCBasicTrigConfig::towAddr2TowNum(int towAddr) {
  
    if (m_TOW_ADDR_2_TOW_NUM[towAddr] == -99 || towAddr < 0 || towAddr > 35){
        throw RPCException("RPCBasicTrigConfig::towAddr2TowNum - wrong towAddr");
        //edm::LogError("RPC")<< "RPCBasicTrigConfig::towAddr2TowNum - wrong towAddr";
    }


  return m_TOW_ADDR_2_TOW_NUM[towAddr];
}

int RPCBasicTrigConfig::getTowerNumOnTb(const RPCConst::l1RpcConeCrdnts& coneCrdnts) {
  return m_TOWER_ON_TB[RPCConst::ITOW_MAX + coneCrdnts.m_Tower];
}

const RPCPacData* RPCBasicTrigConfig::getPac(const RPCConst::l1RpcConeCrdnts& coneCrdnts) const {
  return m_PacManager->getPac(coneCrdnts.m_Tower, coneCrdnts.m_LogSector, coneCrdnts.m_LogSegment);
}

int RPCBasicTrigConfig::getTBNum(const RPCConst::l1RpcConeCrdnts& coneCrdnts) {
  return m_TB_NUM_FOR_TOWER[RPCConst::ITOW_MAX + coneCrdnts.m_Tower];
}
//#############################################################################################
//
//  Constants
//
//#############################################################################################
const int RPCBasicTrigConfig::m_TRIGGER_CRATES_CNT = 12;

const int RPCBasicTrigConfig::m_TOWER_ON_TB[2 * RPCConst::ITOW_MAX + 1 +1] = {
//-16 -15 -14  -13
  0,   1,   2,   3, //tbn4
//-12 -11 -10   -9
  0,   1,   2,   3, //tbn3
//-8  -7   -6   -5
  0,   1,   2,   3, //tbn2
//-4  -3   -2
  0,   1,   2,      //tbn1
//-1   0    1
  0,   1,   2,      //tb0
//2    3    4
  0,   1,   2,      //tbp1
//5    6    7    8
  0,   1,   2,   3, //tbp2
//9   10   11   12
  0,   1,   2,   3, //tbp3
//13  14   15   16
  0,   1,   2,   3, //tbp4
  0            //one more extra
};

const int RPCBasicTrigConfig::m_TOWERS_CNT_ON_TB[m_TB_IN_TC_CNT] = {
  4, 4, 4, 3, 3, 3, 4, 4, 4
};

const int RPCBasicTrigConfig::m_TB_NUM_FOR_TOWER[2 * RPCConst::ITOW_MAX + 1] = {
//-16 -15 -14 -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    0,  0,  0,  0,  1,  1, 1,  1, 2, 2, 2, 2, 3, 3, 3, 4,4,4,5,5,5,6,6,6,6,7, 7, 7, 7, 8, 8, 8, 8
};

const int RPCBasicTrigConfig::m_TOW_ADDR_2_TOW_NUM[36] = {
//0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17	
-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5, -4, -3, -2,-99, -1,  0,
//18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
  1,-99,  2,  3,  4,-99,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16
};
