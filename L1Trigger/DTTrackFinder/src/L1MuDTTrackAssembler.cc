//-------------------------------------------------
//
//   Class: L1MuDTTrackAssembler
//
//   Description: Track Assembler
//
//
//
//   Author :
//   N. Neumeister            CERN EP
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackAssembler.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTTrackFinder/interface/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSectorProcessor.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTExtrapolationUnit.h"

using namespace std;

// --------------------------------
//       class L1MuDTTrackAssembler
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTTrackAssembler::L1MuDTTrackAssembler(const L1MuDTSectorProcessor& sp) : m_sp(sp) {}

//--------------
// Destructor --
//--------------

L1MuDTTrackAssembler::~L1MuDTTrackAssembler() {}

//--------------
// Operations --
//--------------

//
// run Track Assembler
//
void L1MuDTTrackAssembler::run() {
  // get the 18 bitmap tables from the Quality Sorter Unit

  bitset<12> b_adr12_8 = m_sp.EU()->getQSTable(EX12, 0);
  bitset<12> b_adr12_9 = m_sp.EU()->getQSTable(EX12, 1);
  bitset<12> b_adr13_8 = m_sp.EU()->getQSTable(EX13, 0);
  bitset<12> b_adr13_9 = m_sp.EU()->getQSTable(EX13, 1);
  bitset<12> b_adr14_8 = m_sp.EU()->getQSTable(EX14, 0);
  bitset<12> b_adr14_9 = m_sp.EU()->getQSTable(EX14, 1);
  bitset<12> b_adr23_8 = m_sp.EU()->getQSTable(EX23, 0);
  bitset<12> b_adr23_9 = m_sp.EU()->getQSTable(EX23, 1);
  bitset<12> b_adr23_0 = m_sp.EU()->getQSTable(EX23, 2);
  bitset<12> b_adr23_1 = m_sp.EU()->getQSTable(EX23, 3);
  bitset<12> b_adr24_8 = m_sp.EU()->getQSTable(EX24, 0);
  bitset<12> b_adr24_9 = m_sp.EU()->getQSTable(EX24, 1);
  bitset<12> b_adr24_0 = m_sp.EU()->getQSTable(EX24, 2);
  bitset<12> b_adr24_1 = m_sp.EU()->getQSTable(EX24, 3);
  bitset<12> b_adr34_8 = m_sp.EU()->getQSTable(EX34, 0);
  bitset<12> b_adr34_9 = m_sp.EU()->getQSTable(EX34, 1);
  bitset<12> b_adr34_0 = m_sp.EU()->getQSTable(EX34, 2);
  bitset<12> b_adr34_1 = m_sp.EU()->getQSTable(EX34, 3);

  // Last segment node building

  bitset<12> n_1234_888 = (b_adr14_8 & b_adr24_8 & b_adr34_8);
  bitset<12> n_1234_889 = (b_adr14_8 & b_adr24_8 & b_adr34_9);
  bitset<12> n_1234_880 = (b_adr14_8 & b_adr24_8 & b_adr34_0);
  bitset<12> n_1234_881 = (b_adr14_8 & b_adr24_8 & b_adr34_1);
  bitset<12> n_1234_898 = (b_adr14_8 & b_adr24_9 & b_adr34_8);
  bitset<12> n_1234_899 = (b_adr14_8 & b_adr24_9 & b_adr34_9);
  bitset<12> n_1234_890 = (b_adr14_8 & b_adr24_9 & b_adr34_0);
  bitset<12> n_1234_891 = (b_adr14_8 & b_adr24_9 & b_adr34_1);
  bitset<12> n_1234_800 = (b_adr14_8 & b_adr24_0 & b_adr34_0);
  bitset<12> n_1234_801 = (b_adr14_8 & b_adr24_0 & b_adr34_1);
  bitset<12> n_1234_810 = (b_adr14_8 & b_adr24_1 & b_adr34_0);
  bitset<12> n_1234_811 = (b_adr14_8 & b_adr24_1 & b_adr34_1);

  bitset<12> n_1234_988 = (b_adr14_9 & b_adr24_8 & b_adr34_8);
  bitset<12> n_1234_989 = (b_adr14_9 & b_adr24_8 & b_adr34_9);
  bitset<12> n_1234_980 = (b_adr14_9 & b_adr24_8 & b_adr34_0);
  bitset<12> n_1234_981 = (b_adr14_9 & b_adr24_8 & b_adr34_1);
  bitset<12> n_1234_998 = (b_adr14_9 & b_adr24_9 & b_adr34_8);
  bitset<12> n_1234_999 = (b_adr14_9 & b_adr24_9 & b_adr34_9);
  bitset<12> n_1234_990 = (b_adr14_9 & b_adr24_9 & b_adr34_0);
  bitset<12> n_1234_991 = (b_adr14_9 & b_adr24_9 & b_adr34_1);
  bitset<12> n_1234_900 = (b_adr14_9 & b_adr24_0 & b_adr34_0);
  bitset<12> n_1234_901 = (b_adr14_9 & b_adr24_0 & b_adr34_1);
  bitset<12> n_1234_910 = (b_adr14_9 & b_adr24_1 & b_adr34_0);
  bitset<12> n_1234_911 = (b_adr14_9 & b_adr24_1 & b_adr34_1);

  bitset<12> n_123_88 = (b_adr13_8 & b_adr23_8);
  bitset<12> n_123_89 = (b_adr13_8 & b_adr23_9);
  bitset<12> n_123_80 = (b_adr13_8 & b_adr23_0);
  bitset<12> n_123_81 = (b_adr13_8 & b_adr23_1);

  bitset<12> n_123_98 = (b_adr13_9 & b_adr23_8);
  bitset<12> n_123_99 = (b_adr13_9 & b_adr23_9);
  bitset<12> n_123_90 = (b_adr13_9 & b_adr23_0);
  bitset<12> n_123_91 = (b_adr13_9 & b_adr23_1);

  bitset<12> n_124_88 = (b_adr14_8 & b_adr24_8);
  bitset<12> n_124_89 = (b_adr14_8 & b_adr24_9);
  bitset<12> n_124_80 = (b_adr14_8 & b_adr24_0);
  bitset<12> n_124_81 = (b_adr14_8 & b_adr24_1);

  bitset<12> n_124_98 = (b_adr14_9 & b_adr24_8);
  bitset<12> n_124_99 = (b_adr14_9 & b_adr24_9);
  bitset<12> n_124_90 = (b_adr14_9 & b_adr24_0);
  bitset<12> n_124_91 = (b_adr14_9 & b_adr24_1);

  bitset<12> n_134_88 = (b_adr14_8 & b_adr34_8);
  bitset<12> n_134_89 = (b_adr14_8 & b_adr34_9);
  bitset<12> n_134_80 = (b_adr14_8 & b_adr34_0);
  bitset<12> n_134_81 = (b_adr14_8 & b_adr34_1);

  bitset<12> n_134_98 = (b_adr14_9 & b_adr34_8);
  bitset<12> n_134_99 = (b_adr14_9 & b_adr34_9);
  bitset<12> n_134_90 = (b_adr14_9 & b_adr34_0);
  bitset<12> n_134_91 = (b_adr14_9 & b_adr34_1);

  bitset<12> n_234_88 = (b_adr24_8 & b_adr34_8);
  bitset<12> n_234_89 = (b_adr24_8 & b_adr34_9);
  bitset<12> n_234_80 = (b_adr24_8 & b_adr34_0);
  bitset<12> n_234_81 = (b_adr24_8 & b_adr34_1);

  bitset<12> n_234_98 = (b_adr24_9 & b_adr34_8);
  bitset<12> n_234_99 = (b_adr24_9 & b_adr34_9);
  bitset<12> n_234_90 = (b_adr24_9 & b_adr34_0);
  bitset<12> n_234_91 = (b_adr24_9 & b_adr34_1);

  bitset<12> n_12_8 = b_adr12_8;
  bitset<12> n_12_9 = b_adr12_9;

  bitset<12> n_13_8 = b_adr13_8;
  bitset<12> n_13_9 = b_adr13_9;

  bitset<12> n_14_8 = b_adr14_8;
  bitset<12> n_14_9 = b_adr14_9;

  bitset<12> n_23_8 = b_adr23_8;
  bitset<12> n_23_9 = b_adr23_9;

  bitset<12> n_24_8 = b_adr24_8;
  bitset<12> n_24_9 = b_adr24_9;

  bitset<12> n_34_8 = b_adr34_8;
  bitset<12> n_34_9 = b_adr34_9;

  // Last address encoders

  m_theLastAddress[67] = addressEncoder12(n_1234_888);
  m_theLastAddress[66] = addressEncoder12(n_1234_889);
  m_theLastAddress[65] = addressEncoder12(n_1234_880);
  m_theLastAddress[64] = addressEncoder12(n_1234_881);

  m_theLastAddress[63] = addressEncoder12(n_1234_898);
  m_theLastAddress[62] = addressEncoder12(n_1234_899);
  m_theLastAddress[61] = addressEncoder12(n_1234_890);
  m_theLastAddress[60] = addressEncoder12(n_1234_891);

  m_theLastAddress[59] = addressEncoder12(n_1234_800);
  m_theLastAddress[58] = addressEncoder12(n_1234_801);
  m_theLastAddress[57] = addressEncoder12(n_1234_810);
  m_theLastAddress[56] = addressEncoder12(n_1234_811);

  m_theLastAddress[55] = addressEncoder12(n_1234_988);
  m_theLastAddress[54] = addressEncoder12(n_1234_989);
  m_theLastAddress[53] = addressEncoder12(n_1234_980);
  m_theLastAddress[52] = addressEncoder12(n_1234_981);

  m_theLastAddress[51] = addressEncoder12(n_1234_998);
  m_theLastAddress[50] = addressEncoder12(n_1234_999);
  m_theLastAddress[49] = addressEncoder12(n_1234_990);
  m_theLastAddress[48] = addressEncoder12(n_1234_991);

  m_theLastAddress[47] = addressEncoder12(n_1234_900);
  m_theLastAddress[46] = addressEncoder12(n_1234_901);
  m_theLastAddress[45] = addressEncoder12(n_1234_910);
  m_theLastAddress[44] = addressEncoder12(n_1234_911);

  m_theLastAddress[43] = addressEncoder12(n_123_88);
  m_theLastAddress[42] = addressEncoder12(n_123_89);
  m_theLastAddress[41] = addressEncoder12(n_123_80);
  m_theLastAddress[40] = addressEncoder12(n_123_81);

  m_theLastAddress[39] = addressEncoder12(n_123_98);
  m_theLastAddress[38] = addressEncoder12(n_123_99);
  m_theLastAddress[37] = addressEncoder12(n_123_90);
  m_theLastAddress[36] = addressEncoder12(n_123_91);

  m_theLastAddress[35] = addressEncoder12(n_124_88);
  m_theLastAddress[34] = addressEncoder12(n_124_89);
  m_theLastAddress[33] = addressEncoder12(n_124_80);
  m_theLastAddress[32] = addressEncoder12(n_124_81);

  m_theLastAddress[31] = addressEncoder12(n_124_98);
  m_theLastAddress[30] = addressEncoder12(n_124_99);
  m_theLastAddress[29] = addressEncoder12(n_124_90);
  m_theLastAddress[28] = addressEncoder12(n_124_91);

  m_theLastAddress[27] = addressEncoder12(n_134_88);
  m_theLastAddress[26] = addressEncoder12(n_134_89);
  m_theLastAddress[25] = addressEncoder12(n_134_80);
  m_theLastAddress[24] = addressEncoder12(n_134_81);

  m_theLastAddress[23] = addressEncoder12(n_134_98);
  m_theLastAddress[22] = addressEncoder12(n_134_99);
  m_theLastAddress[21] = addressEncoder12(n_134_90);
  m_theLastAddress[20] = addressEncoder12(n_134_91);

  m_theLastAddress[19] = addressEncoder12(n_234_88);
  m_theLastAddress[18] = addressEncoder12(n_234_89);
  m_theLastAddress[17] = addressEncoder12(n_234_80);
  m_theLastAddress[16] = addressEncoder12(n_234_81);

  m_theLastAddress[15] = addressEncoder12(n_234_98);
  m_theLastAddress[14] = addressEncoder12(n_234_99);
  m_theLastAddress[13] = addressEncoder12(n_234_90);
  m_theLastAddress[12] = addressEncoder12(n_234_91);

  m_theLastAddress[11] = addressEncoder12(n_12_8);
  m_theLastAddress[10] = addressEncoder12(n_12_9);

  m_theLastAddress[9] = addressEncoder12(n_13_8);
  m_theLastAddress[8] = addressEncoder12(n_13_9);

  m_theLastAddress[7] = addressEncoder12(n_14_8);
  m_theLastAddress[6] = addressEncoder12(n_14_9);

  m_theLastAddress[5] = addressEncoder12(n_23_8);
  m_theLastAddress[4] = addressEncoder12(n_23_9);

  m_theLastAddress[3] = addressEncoder12(n_24_8);
  m_theLastAddress[2] = addressEncoder12(n_24_9);

  m_theLastAddress[1] = addressEncoder12(n_34_8);
  m_theLastAddress[0] = addressEncoder12(n_34_9);

  m_theLastAddressI[11] = addressEncoder12s(n_12_8);
  m_theLastAddressI[10] = addressEncoder12s(n_12_9);
  m_theLastAddressI[9] = addressEncoder12s(n_13_8);
  m_theLastAddressI[8] = addressEncoder12s(n_13_9);
  m_theLastAddressI[7] = addressEncoder12s(n_14_8);
  m_theLastAddressI[6] = addressEncoder12s(n_14_9);
  m_theLastAddressI[5] = addressEncoder12s(n_23_8);
  m_theLastAddressI[4] = addressEncoder12s(n_23_9);
  m_theLastAddressI[3] = addressEncoder12s(n_24_8);
  m_theLastAddressI[2] = addressEncoder12s(n_24_9);
  m_theLastAddressI[1] = addressEncoder12s(n_34_8);
  m_theLastAddressI[0] = addressEncoder12s(n_34_9);

  // Main equations (68)

  m_thePriorityTable1[67] = (b_adr12_8[0] && b_adr13_8[0] && b_adr23_8[0] && n_1234_888.any());
  m_thePriorityTable1[66] = (b_adr12_8[0] && b_adr13_8[1] && b_adr23_8[1] && n_1234_889.any());
  m_thePriorityTable1[65] = (b_adr12_8[0] && b_adr13_8[2] && b_adr23_8[2] && n_1234_880.any());
  m_thePriorityTable1[64] = (b_adr12_8[0] && b_adr13_8[3] && b_adr23_8[3] && n_1234_881.any());
  m_thePriorityTable1[63] = (b_adr12_8[1] && b_adr13_8[0] && b_adr23_9[0] && n_1234_898.any());
  m_thePriorityTable1[62] = (b_adr12_8[1] && b_adr13_8[1] && b_adr23_9[1] && n_1234_899.any());
  m_thePriorityTable1[61] = (b_adr12_8[1] && b_adr13_8[2] && b_adr23_9[2] && n_1234_890.any());
  m_thePriorityTable1[60] = (b_adr12_8[1] && b_adr13_8[3] && b_adr23_9[3] && n_1234_891.any());
  m_thePriorityTable1[59] = (b_adr12_8[2] && b_adr13_8[2] && b_adr23_0[2] && n_1234_800.any());
  m_thePriorityTable1[58] = (b_adr12_8[2] && b_adr13_8[3] && b_adr23_0[3] && n_1234_801.any());
  m_thePriorityTable1[57] = (b_adr12_8[3] && b_adr13_8[2] && b_adr23_1[2] && n_1234_810.any());
  m_thePriorityTable1[56] = (b_adr12_8[3] && b_adr13_8[3] && b_adr23_1[3] && n_1234_811.any());

  m_thePriorityTable1[55] = (b_adr12_9[0] && b_adr13_9[0] && b_adr23_8[0] && n_1234_988.any());
  m_thePriorityTable1[54] = (b_adr12_9[0] && b_adr13_9[1] && b_adr23_8[1] && n_1234_989.any());
  m_thePriorityTable1[53] = (b_adr12_9[0] && b_adr13_9[2] && b_adr23_8[2] && n_1234_980.any());
  m_thePriorityTable1[52] = (b_adr12_9[0] && b_adr13_9[3] && b_adr23_8[3] && n_1234_981.any());
  m_thePriorityTable1[51] = (b_adr12_9[1] && b_adr13_9[0] && b_adr23_9[0] && n_1234_998.any());
  m_thePriorityTable1[50] = (b_adr12_9[1] && b_adr13_9[1] && b_adr23_9[1] && n_1234_999.any());
  m_thePriorityTable1[49] = (b_adr12_9[1] && b_adr13_9[2] && b_adr23_9[2] && n_1234_990.any());
  m_thePriorityTable1[48] = (b_adr12_9[1] && b_adr13_9[3] && b_adr23_9[3] && n_1234_991.any());
  m_thePriorityTable1[47] = (b_adr12_9[2] && b_adr13_9[2] && b_adr23_0[2] && n_1234_900.any());
  m_thePriorityTable1[46] = (b_adr12_9[2] && b_adr13_9[3] && b_adr23_0[3] && n_1234_901.any());
  m_thePriorityTable1[45] = (b_adr12_9[3] && b_adr13_9[2] && b_adr23_1[2] && n_1234_910.any());
  m_thePriorityTable1[44] = (b_adr12_9[3] && b_adr13_9[3] && b_adr23_1[3] && n_1234_911.any());

  m_thePriorityTable1[43] = (b_adr12_8[0] && n_123_88.any());
  m_thePriorityTable1[42] = (b_adr12_8[1] && n_123_89.any());
  m_thePriorityTable1[41] = (b_adr12_8[2] && n_123_80.any());
  m_thePriorityTable1[40] = (b_adr12_8[3] && n_123_81.any());

  m_thePriorityTable1[39] = (b_adr12_9[0] && n_123_98.any());
  m_thePriorityTable1[38] = (b_adr12_9[1] && n_123_99.any());
  m_thePriorityTable1[37] = (b_adr12_9[2] && n_123_90.any());
  m_thePriorityTable1[36] = (b_adr12_9[3] && n_123_91.any());

  m_thePriorityTable1[35] = (b_adr12_8[0] && n_124_88.any());
  m_thePriorityTable1[34] = (b_adr12_8[1] && n_124_89.any());
  m_thePriorityTable1[33] = (b_adr12_8[2] && n_124_80.any());
  m_thePriorityTable1[32] = (b_adr12_8[3] && n_124_81.any());

  m_thePriorityTable1[31] = (b_adr12_9[0] && n_124_98.any());
  m_thePriorityTable1[30] = (b_adr12_9[1] && n_124_99.any());
  m_thePriorityTable1[29] = (b_adr12_9[2] && n_124_90.any());
  m_thePriorityTable1[28] = (b_adr12_9[3] && n_124_91.any());

  m_thePriorityTable1[27] = (b_adr13_8[0] && n_134_88.any());
  m_thePriorityTable1[26] = (b_adr13_8[1] && n_134_89.any());
  m_thePriorityTable1[25] = (b_adr13_8[2] && n_134_80.any());
  m_thePriorityTable1[24] = (b_adr13_8[3] && n_134_81.any());

  m_thePriorityTable1[23] = (b_adr13_9[0] && n_134_98.any());
  m_thePriorityTable1[22] = (b_adr13_9[1] && n_134_99.any());
  m_thePriorityTable1[21] = (b_adr13_9[2] && n_134_90.any());
  m_thePriorityTable1[20] = (b_adr13_9[3] && n_134_91.any());

  m_thePriorityTable1[19] = (b_adr23_8[0] && n_234_88.any());
  m_thePriorityTable1[18] = (b_adr23_8[1] && n_234_89.any());
  m_thePriorityTable1[17] = (b_adr23_8[2] && n_234_80.any());
  m_thePriorityTable1[16] = (b_adr23_8[3] && n_234_81.any());

  m_thePriorityTable1[15] = (b_adr23_9[0] && n_234_98.any());
  m_thePriorityTable1[14] = (b_adr23_9[1] && n_234_99.any());
  m_thePriorityTable1[13] = (b_adr23_9[2] && n_234_90.any());
  m_thePriorityTable1[12] = (b_adr23_9[3] && n_234_91.any());

  m_thePriorityTable1[11] = n_12_8.any();
  m_thePriorityTable1[10] = n_12_9.any();

  m_thePriorityTable1[9] = n_13_8.any();
  m_thePriorityTable1[8] = n_13_9.any();

  m_thePriorityTable1[7] = n_14_8.any();
  m_thePriorityTable1[6] = n_14_9.any();

  m_thePriorityTable1[5] = n_23_8.any();
  m_thePriorityTable1[4] = n_23_9.any();

  m_thePriorityTable1[3] = n_24_8.any();
  m_thePriorityTable1[2] = n_24_9.any();

  m_thePriorityTable1[1] = n_34_8.any();
  m_thePriorityTable1[0] = n_34_9.any();

  if (!m_thePriorityTable1.any())
    return;

  // first Priority Encoder Sub-Unit
  unsigned int global1 = 0;
  unsigned int group1 = 0;
  unsigned int p1 = 0;
  runEncoderSubUnit1(global1, group1, p1);

  // Address Assignment for the highest priority track
  runAddressAssignment1(global1, group1);

  // Cancellation and second Track Finder Unit
  for (int i = 0; i < 56; i++) {
    m_thePriorityTable2[i] = m_thePriorityTable1[i];
  }
  m_thePriorityTable2 &= getCancelationTable(p1);

  if (!m_thePriorityTable2.any())
    return;

  // second Priority Encoder Sub-Unit
  unsigned int global2 = 0;
  unsigned int group2 = 0;
  unsigned int p2 = 0;
  runEncoderSubUnit2(global2, group2, p2);

  // Address Assignment for the second priority track
  runAddressAssignment2(global2, group2);

  // Fake Pair Cancellation Unit

  unsigned int s1_2 = m_theAddresses[1].station(1);
  unsigned int s2_2 = m_theAddresses[1].station(2);
  unsigned int s3_2 = m_theAddresses[1].station(3);
  unsigned int s4_2 = m_theAddresses[1].station(4);

  if (s2_2 == m_theAddresses[0].station(2)) {
    s2_2 = 15;
    m_theBitMaps[1].reset(1);
    m_theAddresses[1].setStation(2, 15);
    if (m_theTCs[1] == T1234)
      m_theTCs[1] = T134;
    if (m_theTCs[1] == T123)
      m_theTCs[1] = T13;
    if (m_theTCs[1] == T124)
      m_theTCs[1] = T14;
    if (m_theTCs[1] == T234)
      m_theTCs[1] = T34;
  }
  if (s3_2 == m_theAddresses[0].station(3)) {
    s3_2 = 15;
    m_theBitMaps[1].reset(2);
    m_theAddresses[1].setStation(3, 15);
    if (m_theTCs[1] == T1234)
      m_theTCs[1] = T124;
    if (m_theTCs[1] == T123)
      m_theTCs[1] = T12;
    if (m_theTCs[1] == T134)
      m_theTCs[1] = T14;
    if (m_theTCs[1] == T234)
      m_theTCs[1] = T24;
  }
  if (s4_2 == m_theAddresses[0].station(4)) {
    s4_2 = 15;
    m_theBitMaps[1].reset(3);
    m_theAddresses[1].setStation(4, 15);
    if (m_theTCs[1] == T1234)
      m_theTCs[1] = T123;
    if (m_theTCs[1] == T124)
      m_theTCs[1] = T12;
    if (m_theTCs[1] == T134)
      m_theTCs[1] = T13;
    if (m_theTCs[1] == T234)
      m_theTCs[1] = T23;
  }

  if ((s2_2 == 15 && s3_2 == 15 && s4_2 == 15) || (s1_2 == 15 && s3_2 == 15 && s4_2 == 15) ||
      (s1_2 == 15 && s2_2 == 15 && s4_2 == 15) || (s1_2 == 15 && s2_2 == 15 && s3_2 == 15)) {
    if (L1MuDTTFConfig::Debug(5))
      cout << "L1MuDTTrackAssembler: second track has been cancelled" << endl;
    if (L1MuDTTFConfig::Debug(5))
      print();

    m_theTCs[1] = UNDEF;
    m_theAddresses[1].reset();
    m_theBitMaps[1].reset();
  }

  /*
  if ( m_theBitMaps[1].to_ulong() != tc2bitmap(m_theTCs[1]) ) {
    if ( L1MuDTTFConfig::Debug(5) ) cout << "L1MuDTTrackAssembler: second track has been cancelled" << endl;
    if ( L1MuDTTFConfig::Debug(5) ) print();
   
    m_theTCs[1] = UNDEF;
    m_theAddresses[1].reset();
    m_theBitMaps[1].reset();
  }
*/
}

//
// reset Track Assembler
//
void L1MuDTTrackAssembler::reset() {
  for (int i = 0; i < 68; i++)
    m_theLastAddress[i] = 15;
  for (int j = 0; j < 12; j++)
    m_theLastAddressI[j] = 15;
  m_thePriorityTable1.reset();
  m_thePriorityTable2.reset();
  m_theTCs[0] = UNDEF;
  m_theTCs[1] = UNDEF;
  m_theBitMaps[0].reset();
  m_theBitMaps[1].reset();
  m_theAddresses[0].reset();
  m_theAddresses[1].reset();
}

//
// print result of Track Assembler
//
void L1MuDTTrackAssembler::print() const {
  cout << "Track Assembler : " << endl;
  cout << " Priority Table 1 : " << m_thePriorityTable1 << endl;
  cout << " Priority Table 2 : "
       << "            " << m_thePriorityTable2 << endl;

  // print result
  cout << "Track 1: " << m_theTCs[0] << " " << m_theBitMaps[0] << '\t' << m_theAddresses[0] << endl;
  cout << "Track 2: " << m_theTCs[1] << " " << m_theBitMaps[1] << '\t' << m_theAddresses[1] << endl;
}

//
// run the first Priority Encoder Sub-Unit
//
void L1MuDTTrackAssembler::runEncoderSubUnit1(unsigned& global, unsigned& group, unsigned& priority) {
  // Global Grouping

  bitset<22> exi;

  exi[21] = m_thePriorityTable1[67] || m_thePriorityTable1[66] || m_thePriorityTable1[65] || m_thePriorityTable1[64] ||
            m_thePriorityTable1[63] || m_thePriorityTable1[62] || m_thePriorityTable1[61] || m_thePriorityTable1[60] ||
            m_thePriorityTable1[59] || m_thePriorityTable1[58] || m_thePriorityTable1[57] || m_thePriorityTable1[56];
  exi[20] = m_thePriorityTable1[55] || m_thePriorityTable1[54] || m_thePriorityTable1[53] || m_thePriorityTable1[52] ||
            m_thePriorityTable1[51] || m_thePriorityTable1[50] || m_thePriorityTable1[49] || m_thePriorityTable1[48] ||
            m_thePriorityTable1[47] || m_thePriorityTable1[46] || m_thePriorityTable1[45] || m_thePriorityTable1[44];
  exi[19] = m_thePriorityTable1[43] || m_thePriorityTable1[42] || m_thePriorityTable1[41] || m_thePriorityTable1[40];
  exi[18] = m_thePriorityTable1[39] || m_thePriorityTable1[38] || m_thePriorityTable1[37] || m_thePriorityTable1[36];
  exi[17] = m_thePriorityTable1[35] || m_thePriorityTable1[34] || m_thePriorityTable1[33] || m_thePriorityTable1[32];
  exi[16] = m_thePriorityTable1[31] || m_thePriorityTable1[30] || m_thePriorityTable1[29] || m_thePriorityTable1[28];
  exi[15] = m_thePriorityTable1[27] || m_thePriorityTable1[26] || m_thePriorityTable1[25] || m_thePriorityTable1[24];
  exi[14] = m_thePriorityTable1[23] || m_thePriorityTable1[22] || m_thePriorityTable1[21] || m_thePriorityTable1[20];
  exi[13] = m_thePriorityTable1[19] || m_thePriorityTable1[18] || m_thePriorityTable1[17] || m_thePriorityTable1[16];
  exi[12] = m_thePriorityTable1[15] || m_thePriorityTable1[14] || m_thePriorityTable1[13] || m_thePriorityTable1[12];
  exi[11] = m_thePriorityTable1[11];
  exi[10] = m_thePriorityTable1[10];
  exi[9] = m_thePriorityTable1[9];
  exi[8] = m_thePriorityTable1[8];
  exi[7] = m_thePriorityTable1[7];
  exi[6] = m_thePriorityTable1[6];
  exi[5] = m_thePriorityTable1[5];
  exi[4] = m_thePriorityTable1[4];
  exi[3] = m_thePriorityTable1[3];
  exi[2] = m_thePriorityTable1[2];
  exi[1] = m_thePriorityTable1[1];
  exi[0] = m_thePriorityTable1[0];

  // Global Priority Encoder

  global = priorityEncoder22(exi);
  if (global == 31) {
    group = 15;
    priority = 0;
    return;
  }

  // Group priority encoders

  bitset<12> x;
  x = subBitset68(m_thePriorityTable1, 56, 12);
  unsigned int prio1234a = priorityEncoder12(x);
  x = subBitset68(m_thePriorityTable1, 44, 12);
  unsigned int prio1234b = priorityEncoder12(x);

  bitset<4> y;
  y = subBitset68(m_thePriorityTable1, 40, 4);
  unsigned int prio123a = priorityEncoder4(y);
  y = subBitset68(m_thePriorityTable1, 36, 4);
  unsigned int prio123b = priorityEncoder4(y);
  y = subBitset68(m_thePriorityTable1, 32, 4);
  unsigned int prio124a = priorityEncoder4(y);
  y = subBitset68(m_thePriorityTable1, 28, 4);
  unsigned int prio124b = priorityEncoder4(y);
  y = subBitset68(m_thePriorityTable1, 24, 4);
  unsigned int prio134a = priorityEncoder4(y);
  y = subBitset68(m_thePriorityTable1, 20, 4);
  unsigned int prio134b = priorityEncoder4(y);
  y = subBitset68(m_thePriorityTable1, 16, 4);
  unsigned int prio234a = priorityEncoder4(y);
  y = subBitset68(m_thePriorityTable1, 12, 4);
  unsigned int prio234b = priorityEncoder4(y);

  switch (global) {
    case 21: {
      group = prio1234a;
      priority = 56 + group;
      break;
    }
    case 20: {
      group = prio1234b;
      priority = 44 + group;
      break;
    }
    case 19: {
      group = prio123a;
      priority = 40 + group;
      break;
    }
    case 18: {
      group = prio123b;
      priority = 36 + group;
      break;
    }
    case 17: {
      group = prio124a;
      priority = 32 + group;
      break;
    }
    case 16: {
      group = prio124b;
      priority = 28 + group;
      break;
    }
    case 15: {
      group = prio134a;
      priority = 24 + group;
      break;
    }
    case 14: {
      group = prio134b;
      priority = 20 + group;
      break;
    }
    case 13: {
      group = prio234a;
      priority = 16 + group;
      break;
    }
    case 12: {
      group = prio234b;
      priority = 12 + group;
      break;
    }
    default: {
      group = 15;
      priority = global;
      break;
    }
  }
}

//
// run the second Priority Encoder Sub-Unit
//
void L1MuDTTrackAssembler::runEncoderSubUnit2(unsigned& global, unsigned& group, unsigned& priority) {
  // Global Grouping

  bitset<21> exi;

  exi[20] = m_thePriorityTable2[55] || m_thePriorityTable2[54] || m_thePriorityTable2[53] || m_thePriorityTable2[52] ||
            m_thePriorityTable2[51] || m_thePriorityTable2[50] || m_thePriorityTable2[49] || m_thePriorityTable2[48] ||
            m_thePriorityTable2[47] || m_thePriorityTable2[46] || m_thePriorityTable2[45] || m_thePriorityTable2[44];
  exi[19] = m_thePriorityTable2[43] || m_thePriorityTable2[42] || m_thePriorityTable2[41] || m_thePriorityTable2[40];
  exi[18] = m_thePriorityTable2[39] || m_thePriorityTable2[38] || m_thePriorityTable2[37] || m_thePriorityTable2[36];
  exi[17] = m_thePriorityTable2[35] || m_thePriorityTable2[34] || m_thePriorityTable2[33] || m_thePriorityTable2[32];
  exi[16] = m_thePriorityTable2[31] || m_thePriorityTable2[30] || m_thePriorityTable2[29] || m_thePriorityTable2[28];
  exi[15] = m_thePriorityTable2[27] || m_thePriorityTable2[26] || m_thePriorityTable2[25] || m_thePriorityTable2[24];
  exi[14] = m_thePriorityTable2[23] || m_thePriorityTable2[22] || m_thePriorityTable2[21] || m_thePriorityTable2[20];
  exi[13] = m_thePriorityTable2[19] || m_thePriorityTable2[18] || m_thePriorityTable2[17] || m_thePriorityTable2[16];
  exi[12] = m_thePriorityTable2[15] || m_thePriorityTable2[14] || m_thePriorityTable2[13] || m_thePriorityTable2[12];
  exi[11] = m_thePriorityTable2[11];
  exi[10] = m_thePriorityTable2[10];
  exi[9] = m_thePriorityTable2[9];
  exi[8] = m_thePriorityTable2[8];
  exi[7] = m_thePriorityTable2[7];
  exi[6] = m_thePriorityTable2[6];
  exi[5] = m_thePriorityTable2[5];
  exi[4] = m_thePriorityTable2[4];
  exi[3] = m_thePriorityTable2[3];
  exi[2] = m_thePriorityTable2[2];
  exi[1] = m_thePriorityTable2[1];
  exi[0] = m_thePriorityTable2[0];

  // Global Priority Encoder

  global = priorityEncoder21(exi);
  if (global == 31) {
    group = 15;
    priority = 0;
    return;
  }

  // Group priority encoders

  bitset<12> x;
  x = subBitset56(m_thePriorityTable2, 44, 12);
  unsigned int prio1234b = priorityEncoder12(x);

  bitset<4> y;
  y = subBitset56(m_thePriorityTable2, 40, 4);
  unsigned int prio123a = priorityEncoder4(y);
  y = subBitset56(m_thePriorityTable2, 36, 4);
  unsigned int prio123b = priorityEncoder4(y);
  y = subBitset56(m_thePriorityTable2, 32, 4);
  unsigned int prio124a = priorityEncoder4(y);
  y = subBitset56(m_thePriorityTable2, 28, 4);
  unsigned int prio124b = priorityEncoder4(y);
  y = subBitset56(m_thePriorityTable2, 24, 4);
  unsigned int prio134a = priorityEncoder4(y);
  y = subBitset56(m_thePriorityTable2, 20, 4);
  unsigned int prio134b = priorityEncoder4(y);
  y = subBitset56(m_thePriorityTable2, 16, 4);
  unsigned int prio234a = priorityEncoder4(y);
  y = subBitset56(m_thePriorityTable2, 12, 4);
  unsigned int prio234b = priorityEncoder4(y);

  switch (global) {
    case 20: {
      group = prio1234b;
      priority = 44 + group;
      break;
    }
    case 19: {
      group = prio123a;
      priority = 40 + group;
      break;
    }
    case 18: {
      group = prio123b;
      priority = 36 + group;
      break;
    }
    case 17: {
      group = prio124a;
      priority = 32 + group;
      break;
    }
    case 16: {
      group = prio124b;
      priority = 28 + group;
      break;
    }
    case 15: {
      group = prio134a;
      priority = 24 + group;
      break;
    }
    case 14: {
      group = prio134b;
      priority = 20 + group;
      break;
    }
    case 13: {
      group = prio234a;
      priority = 16 + group;
      break;
    }
    case 12: {
      group = prio234b;
      priority = 12 + group;
      break;
    }
    default: {
      group = 15;
      priority = global;
      break;
    }
  }
}

//
// run the first Address Assignment Sub-Unit
//
void L1MuDTTrackAssembler::runAddressAssignment1(int global, int group) {
  TrackClass tc(UNDEF);

  switch (global) {
    case 21: {
      tc = T1234;
      switch (group) {
        case 11:
          m_theAddresses[0].setStations(0, 0, 0, m_theLastAddress[67]);
          break;
        case 10:
          m_theAddresses[0].setStations(0, 0, 1, m_theLastAddress[66]);
          break;
        case 9:
          m_theAddresses[0].setStations(0, 0, 2, m_theLastAddress[65]);
          break;
        case 8:
          m_theAddresses[0].setStations(0, 0, 3, m_theLastAddress[64]);
          break;
        case 7:
          m_theAddresses[0].setStations(0, 1, 0, m_theLastAddress[63]);
          break;
        case 6:
          m_theAddresses[0].setStations(0, 1, 1, m_theLastAddress[62]);
          break;
        case 5:
          m_theAddresses[0].setStations(0, 1, 2, m_theLastAddress[61]);
          break;
        case 4:
          m_theAddresses[0].setStations(0, 1, 3, m_theLastAddress[60]);
          break;
        case 3:
          m_theAddresses[0].setStations(0, 2, 2, m_theLastAddress[59]);
          break;
        case 2:
          m_theAddresses[0].setStations(0, 2, 3, m_theLastAddress[58]);
          break;
        case 1:
          m_theAddresses[0].setStations(0, 3, 2, m_theLastAddress[57]);
          break;
        case 0:
          m_theAddresses[0].setStations(0, 3, 3, m_theLastAddress[56]);
          break;
      }
      break;
    }
    case 20: {
      tc = T1234;
      switch (group) {
        case 11:
          m_theAddresses[0].setStations(1, 0, 0, m_theLastAddress[55]);
          break;
        case 10:
          m_theAddresses[0].setStations(1, 0, 1, m_theLastAddress[54]);
          break;
        case 9:
          m_theAddresses[0].setStations(1, 0, 2, m_theLastAddress[53]);
          break;
        case 8:
          m_theAddresses[0].setStations(1, 0, 3, m_theLastAddress[52]);
          break;
        case 7:
          m_theAddresses[0].setStations(1, 1, 0, m_theLastAddress[51]);
          break;
        case 6:
          m_theAddresses[0].setStations(1, 1, 1, m_theLastAddress[50]);
          break;
        case 5:
          m_theAddresses[0].setStations(1, 1, 2, m_theLastAddress[49]);
          break;
        case 4:
          m_theAddresses[0].setStations(1, 1, 3, m_theLastAddress[48]);
          break;
        case 3:
          m_theAddresses[0].setStations(1, 2, 2, m_theLastAddress[47]);
          break;
        case 2:
          m_theAddresses[0].setStations(1, 2, 3, m_theLastAddress[46]);
          break;
        case 1:
          m_theAddresses[0].setStations(1, 3, 2, m_theLastAddress[45]);
          break;
        case 0:
          m_theAddresses[0].setStations(1, 3, 3, m_theLastAddress[44]);
          break;
      }
      break;
    }
    case 19: {
      tc = T123;
      switch (group) {
        case 3:
          m_theAddresses[0].setStations(0, 0, m_theLastAddress[43], 15);
          break;
        case 2:
          m_theAddresses[0].setStations(0, 1, m_theLastAddress[42], 15);
          break;
        case 1:
          m_theAddresses[0].setStations(0, 2, m_theLastAddress[41], 15);
          break;
        case 0:
          m_theAddresses[0].setStations(0, 3, m_theLastAddress[40], 15);
          break;
      }
      break;
    }
    case 18: {
      tc = T123;
      switch (group) {
        case 3:
          m_theAddresses[0].setStations(1, 0, m_theLastAddress[39], 15);
          break;
        case 2:
          m_theAddresses[0].setStations(1, 1, m_theLastAddress[38], 15);
          break;
        case 1:
          m_theAddresses[0].setStations(1, 2, m_theLastAddress[37], 15);
          break;
        case 0:
          m_theAddresses[0].setStations(1, 3, m_theLastAddress[36], 15);
          break;
      }
      break;
    }
    case 17: {
      tc = T124;
      switch (group) {
        case 3:
          m_theAddresses[0].setStations(0, 0, 15, m_theLastAddress[35]);
          break;
        case 2:
          m_theAddresses[0].setStations(0, 1, 15, m_theLastAddress[34]);
          break;
        case 1:
          m_theAddresses[0].setStations(0, 2, 15, m_theLastAddress[33]);
          break;
        case 0:
          m_theAddresses[0].setStations(0, 3, 15, m_theLastAddress[32]);
          break;
      }
      break;
    }
    case 16: {
      tc = T124;
      switch (group) {
        case 3:
          m_theAddresses[0].setStations(1, 0, 15, m_theLastAddress[31]);
          break;
        case 2:
          m_theAddresses[0].setStations(1, 1, 15, m_theLastAddress[30]);
          break;
        case 1:
          m_theAddresses[0].setStations(1, 2, 15, m_theLastAddress[29]);
          break;
        case 0:
          m_theAddresses[0].setStations(1, 3, 15, m_theLastAddress[28]);
          break;
      }
      break;
    }
    case 15: {
      tc = T134;
      switch (group) {
        case 3:
          m_theAddresses[0].setStations(0, 15, 0, m_theLastAddress[27]);
          break;
        case 2:
          m_theAddresses[0].setStations(0, 15, 1, m_theLastAddress[26]);
          break;
        case 1:
          m_theAddresses[0].setStations(0, 15, 2, m_theLastAddress[25]);
          break;
        case 0:
          m_theAddresses[0].setStations(0, 15, 3, m_theLastAddress[24]);
          break;
      }
      break;
    }
    case 14: {
      tc = T134;
      switch (group) {
        case 3:
          m_theAddresses[0].setStations(1, 15, 0, m_theLastAddress[23]);
          break;
        case 2:
          m_theAddresses[0].setStations(1, 15, 1, m_theLastAddress[22]);
          break;
        case 1:
          m_theAddresses[0].setStations(1, 15, 2, m_theLastAddress[21]);
          break;
        case 0:
          m_theAddresses[0].setStations(1, 15, 3, m_theLastAddress[20]);
          break;
      }
      break;
    }
    case 13: {
      tc = T234;
      switch (group) {
        case 3:
          m_theAddresses[0].setStations(15, 0, 0, m_theLastAddress[19]);
          break;
        case 2:
          m_theAddresses[0].setStations(15, 0, 1, m_theLastAddress[18]);
          break;
        case 1:
          m_theAddresses[0].setStations(15, 0, 2, m_theLastAddress[17]);
          break;
        case 0:
          m_theAddresses[0].setStations(15, 0, 3, m_theLastAddress[16]);
          break;
      }
      break;
    }
    case 12: {
      tc = T234;
      switch (group) {
        case 3:
          m_theAddresses[0].setStations(15, 1, 0, m_theLastAddress[15]);
          break;
        case 2:
          m_theAddresses[0].setStations(15, 1, 1, m_theLastAddress[14]);
          break;
        case 1:
          m_theAddresses[0].setStations(15, 1, 2, m_theLastAddress[13]);
          break;
        case 0:
          m_theAddresses[0].setStations(15, 1, 3, m_theLastAddress[12]);
          break;
      }
      break;
    }
    case 11: {
      tc = T12;
      m_theAddresses[0].setStations(0, m_theLastAddress[11], 15, 15);
      break;
    }
    case 10: {
      tc = T12;
      m_theAddresses[0].setStations(1, m_theLastAddress[10], 15, 15);
      break;
    }
    case 9: {
      tc = T13;
      m_theAddresses[0].setStations(0, 15, m_theLastAddress[9], 15);
      break;
    }
    case 8: {
      tc = T13;
      m_theAddresses[0].setStations(1, 15, m_theLastAddress[8], 15);
      break;
    }
    case 7: {
      tc = T14;
      m_theAddresses[0].setStations(0, 15, 15, m_theLastAddress[7]);
      break;
    }
    case 6: {
      tc = T14;
      m_theAddresses[0].setStations(1, 15, 15, m_theLastAddress[6]);
      break;
    }
    case 5: {
      tc = T23;
      m_theAddresses[0].setStations(15, 0, m_theLastAddress[5], 15);
      break;
    }
    case 4: {
      tc = T23;
      m_theAddresses[0].setStations(15, 1, m_theLastAddress[4], 15);
      break;
    }
    case 3: {
      tc = T24;
      m_theAddresses[0].setStations(15, 0, 15, m_theLastAddress[3]);
      break;
    }
    case 2: {
      tc = T24;
      m_theAddresses[0].setStations(15, 1, 15, m_theLastAddress[2]);
      break;
    }
    case 1: {
      tc = T34;
      m_theAddresses[0].setStations(15, 15, 0, m_theLastAddress[1]);
      break;
    }
    case 0: {
      tc = T34;
      m_theAddresses[0].setStations(15, 15, 1, m_theLastAddress[0]);
      break;
    }
  }

  // set Track Class and covert to bitmap
  m_theTCs[0] = tc;
  m_theBitMaps[0] = tc2bitmap(tc);
}

//
// run the second Address Assignment Sub-Unit
//
void L1MuDTTrackAssembler::runAddressAssignment2(int global, int group) {
  TrackClass tc(UNDEF);

  switch (global) {
    case 20: {
      tc = T1234;
      switch (group) {
        case 11:
          m_theAddresses[1].setStations(1, 0, 0, m_theLastAddress[55]);
          break;
        case 10:
          m_theAddresses[1].setStations(1, 0, 1, m_theLastAddress[54]);
          break;
        case 9:
          m_theAddresses[1].setStations(1, 0, 2, m_theLastAddress[53]);
          break;
        case 8:
          m_theAddresses[1].setStations(1, 0, 3, m_theLastAddress[52]);
          break;
        case 7:
          m_theAddresses[1].setStations(1, 1, 0, m_theLastAddress[51]);
          break;
        case 6:
          m_theAddresses[1].setStations(1, 1, 1, m_theLastAddress[50]);
          break;
        case 5:
          m_theAddresses[1].setStations(1, 1, 2, m_theLastAddress[49]);
          break;
        case 4:
          m_theAddresses[1].setStations(1, 1, 3, m_theLastAddress[48]);
          break;
        case 3:
          m_theAddresses[1].setStations(1, 2, 2, m_theLastAddress[47]);
          break;
        case 2:
          m_theAddresses[1].setStations(1, 2, 3, m_theLastAddress[46]);
          break;
        case 1:
          m_theAddresses[1].setStations(1, 3, 2, m_theLastAddress[45]);
          break;
        case 0:
          m_theAddresses[1].setStations(1, 3, 3, m_theLastAddress[44]);
          break;
      }
      break;
    }
    case 19: {
      tc = T123;
      switch (group) {
        case 3:
          m_theAddresses[1].setStations(0, 0, m_theLastAddress[43], 15);
          break;
        case 2:
          m_theAddresses[1].setStations(0, 1, m_theLastAddress[42], 15);
          break;
        case 1:
          m_theAddresses[1].setStations(0, 2, m_theLastAddress[41], 15);
          break;
        case 0:
          m_theAddresses[1].setStations(0, 3, m_theLastAddress[40], 15);
          break;
      }
      break;
    }
    case 18: {
      tc = T123;
      switch (group) {
        case 3:
          m_theAddresses[1].setStations(1, 0, m_theLastAddress[39], 15);
          break;
        case 2:
          m_theAddresses[1].setStations(1, 1, m_theLastAddress[38], 15);
          break;
        case 1:
          m_theAddresses[1].setStations(1, 2, m_theLastAddress[37], 15);
          break;
        case 0:
          m_theAddresses[1].setStations(1, 3, m_theLastAddress[36], 15);
          break;
      }
      break;
    }
    case 17: {
      tc = T124;
      switch (group) {
        case 3:
          m_theAddresses[1].setStations(0, 0, 15, m_theLastAddress[35]);
          break;
        case 2:
          m_theAddresses[1].setStations(0, 1, 15, m_theLastAddress[34]);
          break;
        case 1:
          m_theAddresses[1].setStations(0, 2, 15, m_theLastAddress[33]);
          break;
        case 0:
          m_theAddresses[1].setStations(0, 3, 15, m_theLastAddress[32]);
          break;
      }
      break;
    }
    case 16: {
      tc = T124;
      switch (group) {
        case 3:
          m_theAddresses[1].setStations(1, 0, 15, m_theLastAddress[31]);
          break;
        case 2:
          m_theAddresses[1].setStations(1, 1, 15, m_theLastAddress[30]);
          break;
        case 1:
          m_theAddresses[1].setStations(1, 2, 15, m_theLastAddress[29]);
          break;
        case 0:
          m_theAddresses[1].setStations(1, 3, 15, m_theLastAddress[28]);
          break;
      }
      break;
    }
    case 15: {
      tc = T134;
      switch (group) {
        case 3:
          m_theAddresses[1].setStations(0, 15, 0, m_theLastAddress[27]);
          break;
        case 2:
          m_theAddresses[1].setStations(0, 15, 1, m_theLastAddress[26]);
          break;
        case 1:
          m_theAddresses[1].setStations(0, 15, 2, m_theLastAddress[25]);
          break;
        case 0:
          m_theAddresses[1].setStations(0, 15, 3, m_theLastAddress[24]);
          break;
      }
      break;
    }
    case 14: {
      tc = T134;
      switch (group) {
        case 3:
          m_theAddresses[1].setStations(1, 15, 0, m_theLastAddress[23]);
          break;
        case 2:
          m_theAddresses[1].setStations(1, 15, 1, m_theLastAddress[22]);
          break;
        case 1:
          m_theAddresses[1].setStations(1, 15, 2, m_theLastAddress[21]);
          break;
        case 0:
          m_theAddresses[1].setStations(1, 15, 3, m_theLastAddress[20]);
          break;
      }
      break;
    }
    case 13: {
      tc = T234;
      switch (group) {
        case 3:
          m_theAddresses[1].setStations(15, 0, 0, m_theLastAddress[19]);
          break;
        case 2:
          m_theAddresses[1].setStations(15, 0, 1, m_theLastAddress[18]);
          break;
        case 1:
          m_theAddresses[1].setStations(15, 0, 2, m_theLastAddress[17]);
          break;
        case 0:
          m_theAddresses[1].setStations(15, 0, 3, m_theLastAddress[16]);
          break;
      }
      break;
    }
    case 12: {
      tc = T234;
      switch (group) {
        case 3:
          m_theAddresses[1].setStations(15, 1, 0, m_theLastAddress[15]);
          break;
        case 2:
          m_theAddresses[1].setStations(15, 1, 1, m_theLastAddress[14]);
          break;
        case 1:
          m_theAddresses[1].setStations(15, 1, 2, m_theLastAddress[13]);
          break;
        case 0:
          m_theAddresses[1].setStations(15, 1, 3, m_theLastAddress[12]);
          break;
      }
      break;
    }
    case 11: {
      tc = T12;
      m_theAddresses[1].setStations(0, m_theLastAddressI[11], 15, 15);
      break;
    }
    case 10: {
      tc = T12;
      m_theAddresses[1].setStations(1, m_theLastAddressI[10], 15, 15);
      break;
    }
    case 9: {
      tc = T13;
      m_theAddresses[1].setStations(0, 15, m_theLastAddressI[9], 15);
      break;
    }
    case 8: {
      tc = T13;
      m_theAddresses[1].setStations(1, 15, m_theLastAddressI[8], 15);
      break;
    }
    case 7: {
      tc = T14;
      m_theAddresses[1].setStations(0, 15, 15, m_theLastAddressI[7]);
      break;
    }
    case 6: {
      tc = T14;
      m_theAddresses[1].setStations(1, 15, 15, m_theLastAddressI[6]);
      break;
    }
    case 5: {
      tc = T23;
      m_theAddresses[1].setStations(15, 0, m_theLastAddressI[5], 15);
      break;
    }
    case 4: {
      tc = T23;
      m_theAddresses[1].setStations(15, 1, m_theLastAddressI[4], 15);
      break;
    }
    case 3: {
      tc = T24;
      m_theAddresses[1].setStations(15, 0, 15, m_theLastAddressI[3]);
      break;
    }
    case 2: {
      tc = T24;
      m_theAddresses[1].setStations(15, 1, 15, m_theLastAddressI[2]);
      break;
    }
    case 1: {
      tc = T34;
      m_theAddresses[1].setStations(15, 15, 0, m_theLastAddressI[1]);
      break;
    }
    case 0: {
      tc = T34;
      m_theAddresses[1].setStations(15, 15, 1, m_theLastAddressI[0]);
      break;
    }
  }

  // set Track Class and covert to bitmap
  m_theTCs[1] = tc;
  m_theBitMaps[1] = tc2bitmap(tc);
}

//
// 12 bit priority encoder
//
unsigned int L1MuDTTrackAssembler::priorityEncoder12(const bitset<12>& input) {
  unsigned int result = 15;

  for (int i = 0; i < 12; i++) {
    if (input.test(i))
      result = i;
  }

  return result;
}

//
// 4 bit priority encoder
//
unsigned int L1MuDTTrackAssembler::priorityEncoder4(const bitset<4>& input) {
  unsigned int result = 3;

  for (int i = 0; i < 4; i++) {
    if (input.test(i))
      result = i;
  }

  return result;
}

//
// 22 bit priority encoder
//
unsigned int L1MuDTTrackAssembler::priorityEncoder22(const bitset<22>& input) {
  unsigned int result = 31;

  for (int i = 0; i < 22; i++) {
    if (input.test(i))
      result = i;
  }

  return result;
}

//
// 21 bit priority encoder
//
unsigned int L1MuDTTrackAssembler::priorityEncoder21(const bitset<21>& input) {
  unsigned int result = 31;

  for (int i = 0; i < 21; i++) {
    if (input.test(i))
      result = i;
  }

  return result;
}

//
// 12 bit address encoder
//
unsigned int L1MuDTTrackAssembler::addressEncoder12(const bitset<12>& input) {
  // inverse order priority encoder

  unsigned int result = 15;

  for (int i = 0; i < 12; i++) {
    if (input.test(i)) {
      result = i;
      break;
    }
  }

  return result;
}

//
// special 12 bit address encoder
//
unsigned int L1MuDTTrackAssembler::addressEncoder12s(const bitset<12>& input) {
  // inverse order priority encoder which prefers second addresses

  unsigned int result = 15;

  for (int i = 0; i < 11; i += 2) {
    if (input.test(i) || input.test(i + 1)) {
      if (input.test(i))
        result = i;
      if (input.test(i + 1))
        result = i + 1;
      break;
    }
  }

  return result;
}

//
// get sub-bitmap of a 68-bit word
//
unsigned long L1MuDTTrackAssembler::subBitset68(const bitset<68>& input, int pos, int length) {
  bitset<68> s(input);

  for (int i = pos + length; i < 68; i++)
    s.reset(i);

  s >>= pos;

  return s.to_ulong();
}

//
// get sub-bitmap of a 56-bit word
//
unsigned long L1MuDTTrackAssembler::subBitset56(const bitset<56>& input, int pos, int length) {
  bitset<56> s(input);

  for (int i = pos + length; i < 56; i++)
    s.reset(i);

  s >>= pos;

  return s.to_ulong();
}

//
// Cancel Out Table
//
bitset<56> L1MuDTTrackAssembler::getCancelationTable(unsigned int p) {
  // Cancellation Table
  // Each 0 in this Table means a sub-track of the
  // previous found Track

  switch (p) {
    case 67: {
      bitset<56> b(string("00000111111100000111000001110000011100000111010101010101"));
      return b;
      break;
    }
    case 66: {
      bitset<56> b(string("00001011111100000111000001110000101100001011010101010110"));
      return b;
      break;
    }
    case 65: {
      bitset<56> b(string("00001101010100000111000001110000110100001101010101010111"));
      return b;
      break;
    }
    case 64: {
      bitset<56> b(string("00001110101000000111000001110000111000001110010101010111"));
      return b;
      break;
    }
    case 63: {
      bitset<56> b(string("01110000111100001011000010110000011101110000010101101001"));
      return b;
      break;
    }
    case 62: {
      bitset<56> b(string("10110000111100001011000010110000101110110000010101101010"));
      return b;
      break;
    }
    case 61: {
      bitset<56> b(string("11010000111100001011000010110000110111010000010101101011"));
      return b;
      break;
    }
    case 60: {
      bitset<56> b(string("11100000111100001011000010110000111011100000010101101011"));
      return b;
      break;
    }
    case 59: {
      bitset<56> b(string("11011101000100001101000011010000110111011101010101111111"));
      return b;
      break;
    }
    case 58: {
      bitset<56> b(string("11101110001000001101000011010000111011101110010101111111"));
      return b;
      break;
    }
    case 57: {
      bitset<56> b(string("11011101010000001110000011100000110111011101010101111111"));
      return b;
      break;
    }
    case 56: {
      bitset<56> b(string("11101110100000001110000011100000111011101110010101111111"));
      return b;
      break;
    }

    case 55: {
      bitset<56> b(string("00000000000001110000011100000111000000000111101010010101"));
      return b;
      break;
    }
    case 54: {
      bitset<56> b(string("00000000000001110000011100001011000000001011101010010110"));
      return b;
      break;
    }
    case 53: {
      bitset<56> b(string("00000000000001110000011100001101000000001101101010010111"));
      return b;
      break;
    }
    case 52: {
      bitset<56> b(string("00000000000001110000011100001110000000001110101010010111"));
      return b;
      break;
    }
    case 51: {
      bitset<56> b(string("00000000000010110000101100000111000001110000101010101001"));
      return b;
      break;
    }
    case 50: {
      bitset<56> b(string("00000000000010110000101100001011000010110000101010101010"));
      return b;
      break;
    }
    case 49: {
      bitset<56> b(string("00000000000010110000101100001101000011010000101010101011"));
      return b;
      break;
    }
    case 48: {
      bitset<56> b(string("00000000000010110000101100001110000011100000101010101011"));
      return b;
      break;
    }
    case 47: {
      bitset<56> b(string("00000000000011010000110100001101000011011101101010111111"));
      return b;
      break;
    }
    case 46: {
      bitset<56> b(string("00000000000011010000110100001110000011101110101010111111"));
      return b;
      break;
    }
    case 45: {
      bitset<56> b(string("00000000000011100000111000001101000011011101101010111111"));
      return b;
      break;
    }
    case 44: {
      bitset<56> b(string("00000000000011100000111000001110000011101110101010111111"));
      return b;
      break;
    }

    case 43: {
      bitset<56> b(string("00000000000000000111000001110000111100001111010101010111"));
      return b;
      break;
    }
    case 42: {
      bitset<56> b(string("00000000000000001011000010110000111111110000010101101011"));
      return b;
      break;
    }
    case 41: {
      bitset<56> b(string("00000000000000001101000011010000111111111111010101111111"));
      return b;
      break;
    }
    case 40: {
      bitset<56> b(string("00000000000000001110000011100000111111111111010101111111"));
      return b;
      break;
    }

    case 39: {
      bitset<56> b(string("00000000000000000000011100001111000000001111101010010111"));
      return b;
      break;
    }
    case 38: {
      bitset<56> b(string("00000000000000000000101100001111000011110000101010101011"));
      return b;
      break;
    }
    case 37: {
      bitset<56> b(string("00000000000000000000110100001111000011111111101010111111"));
      return b;
      break;
    }
    case 36: {
      bitset<56> b(string("00000000000000000000111000001111000011111111101010111111"));
      return b;
      break;
    }

    case 35: {
      bitset<56> b(string("00000000000000000000000001110000111100001111010101010111"));
      return b;
      break;
    }
    case 34: {
      bitset<56> b(string("00000000000000000000000010110000111111110000010101101011"));
      return b;
      break;
    }
    case 33: {
      bitset<56> b(string("00000000000000000000000011010000111111111111010101111111"));
      return b;
      break;
    }
    case 32: {
      bitset<56> b(string("00000000000000000000000011100000111111111111010101111111"));
      return b;
      break;
    }

    case 31: {
      bitset<56> b(string("00000000000000000000011100001111000000001111101010010111"));
      return b;
      break;
    }
    case 30: {
      bitset<56> b(string("00000000000000000000101100001111000011110000101010101011"));
      return b;
      break;
    }
    case 29: {
      bitset<56> b(string("00000000000000000000110100001111000011111111101010111111"));
      return b;
      break;
    }
    case 28: {
      bitset<56> b(string("00000000000000000000111000001111000011111111101010111111"));
      return b;
      break;
    }

    case 27: {
      bitset<56> b(string("00000000000000000000000000000000011101110111010101111101"));
      return b;
      break;
    }
    case 26: {
      bitset<56> b(string("00000000000000000000000000000000101110111011010101111110"));
      return b;
      break;
    }
    case 25: {
      bitset<56> b(string("00000000000000000000000000000000110111011101010101111111"));
      return b;
      break;
    }
    case 24: {
      bitset<56> b(string("00000000000000000000000000000000111011101110010101111111"));
      return b;
      break;
    }

    case 23: {
      bitset<56> b(string("00000000000000000000000000000000000001110111101010111101"));
      return b;
      break;
    }
    case 22: {
      bitset<56> b(string("00000000000000000000000000000000000010111011101010111110"));
      return b;
      break;
    }
    case 21: {
      bitset<56> b(string("00000000000000000000000000000000000011011101101010111111"));
      return b;
      break;
    }
    case 20: {
      bitset<56> b(string("00000000000000000000000000000000000011101110101010111111"));
      return b;
      break;
    }

    case 19: {
      bitset<56> b(string("00000000000000000000000000000000000000000111111111010101"));
      return b;
      break;
    }
    case 18: {
      bitset<56> b(string("00000000000000000000000000000000000000001011111111010110"));
      return b;
      break;
    }
    case 17: {
      bitset<56> b(string("00000000000000000000000000000000000000001101111111010111"));
      return b;
      break;
    }
    case 16: {
      bitset<56> b(string("00000000000000000000000000000000000000001110111111010111"));
      return b;
      break;
    }

    case 15: {
      bitset<56> b(string("00000000000000000000000000000000000000000000111111101001"));
      return b;
      break;
    }
    case 14: {
      bitset<56> b(string("00000000000000000000000000000000000000000000111111101010"));
      return b;
      break;
    }
    case 13: {
      bitset<56> b(string("00000000000000000000000000000000000000000000111111101011"));
      return b;
      break;
    }
    case 12: {
      bitset<56> b(string("00000000000000000000000000000000000000000000111111101011"));
      return b;
      break;
    }

    case 11: {
      bitset<56> b(string("00000000000000000000000000000000000000000000010101111111"));
      return b;
      break;
    }
    case 10: {
      bitset<56> b(string("00000000000000000000000000000000000000000000001010111111"));
      return b;
      break;
    }

    case 9: {
      bitset<56> b(string("00000000000000000000000000000000000000000000000101111111"));
      return b;
      break;
    }
    case 8: {
      bitset<56> b(string("00000000000000000000000000000000000000000000000010111111"));
      return b;
      break;
    }

    case 7: {
      bitset<56> b(string("00000000000000000000000000000000000000000000000001111111"));
      return b;
      break;
    }
    case 6: {
      bitset<56> b(string("00000000000000000000000000000000000000000000000000111111"));
      return b;
      break;
    }

    case 5: {
      bitset<56> b(string("00000000000000000000000000000000000000000000000000010111"));
      return b;
      break;
    }
    case 4: {
      bitset<56> b(string("00000000000000000000000000000000000000000000000000001011"));
      return b;
      break;
    }

    case 3: {
      bitset<56> b(string("00000000000000000000000000000000000000000000000000000111"));
      return b;
      break;
    }
    case 2: {
      bitset<56> b(string("00000000000000000000000000000000000000000000000000000011"));
      return b;
      break;
    }

    case 1: {
      bitset<56> b(string("00000000000000000000000000000000000000000000000000000001"));
      return b;
      break;
    }
    case 0: {
      bitset<56> b(string("00000000000000000000000000000000000000000000000000000000"));
      return b;
      break;
    }
    default: {
      bitset<56> b;
      return b;
      break;
    }
  }
}
