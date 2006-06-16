//---------------------------------------------------------------------------

#include "L1Trigger/RPCTrigger/src/L1RpcBasicTrigConfig.h"

//---------------------------------------------------------------------------

int L1RpcBasicTrigConfig::GetTowerNumOnTb(const RPCParam::L1RpcConeCrdnts& coneCrdnts) {
  return TOWER_ON_TB[L1RpcConst::ITOW_MAX + coneCrdnts.Tower];
}

const L1RpcPac* L1RpcBasicTrigConfig::GetPac(const RPCParam::L1RpcConeCrdnts& coneCrdnts) const {
  return PacManager->GetPac(coneCrdnts.Tower, coneCrdnts.LogSector, coneCrdnts.LogSegment);
}

int L1RpcBasicTrigConfig::GetTBNum(const RPCParam::L1RpcConeCrdnts& coneCrdnts) {
  return TB_NUM_FOR_TOWER[L1RpcConst::ITOW_MAX + coneCrdnts.Tower];
}

const int L1RpcBasicTrigConfig::TRIGGER_CRATES_CNT = 12;

const int L1RpcBasicTrigConfig::TOWER_ON_TB[2 * L1RpcConst::ITOW_MAX + 1 +1] = {
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

const int L1RpcBasicTrigConfig::TOWERS_CNT_ON_TB[TB_IN_TC_CNT] = {
  4, 4, 4, 3, 3, 3, 4, 4, 4
};

const int L1RpcBasicTrigConfig::TB_NUM_FOR_TOWER[2 * L1RpcConst::ITOW_MAX + 1] = {
//-16 -15 -14 -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    0,  0,  0,  0,  1,  1, 1,  1, 2, 2, 2, 2, 3, 3, 3, 4,4,4,5,5,5,6,6,6,6,7, 7, 7, 7, 8, 8, 8, 8
};

const int L1RpcBasicTrigConfig::TOW_ADDR_2_TOW_NUM[36] = {
//0	  1	  2	  3	  4	  5	  6	 7	8	 9 10	11	12	13	14	15	16	17	18	19	20	21	22	23	24	25	26	27	28	29	30	31	32	33	34	35
-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5, -4, -3, -2,-99, -1,  0,  1,-99,  2,  3,  4,-99,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16
};
