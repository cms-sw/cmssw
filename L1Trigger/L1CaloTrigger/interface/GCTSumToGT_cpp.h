//------------------------------------
// More logic for Phase2L1GCTSumEmulator GT output
// (based heavily on algo_top.cpp in TO_GT_IP firmware repo)
//------------------------------------
#ifndef L1Trigger_L1CaloTrigger_GCTSumToGT_cpp
#define L1Trigger_L1CaloTrigger_GCTSumToGT_cpp

#include <ap_int.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "L1Trigger/L1CaloTrigger/interface/GCTSumToGT_h.h"

namespace p2gctsumGT {

inline void processOutLinks_GT(ap_uint<576> link_out_reg[N_OUTPUT_LINKS_GT], ap_uint<576> link_out[N_OUTPUT_LINKS_GT]) {
  for (loop idx = 0; idx < N_OUTPUT_LINKS_GT; idx++) {
    link_out[idx].range(575, 0) = link_out_reg[idx].range(575, 0);
  }
}

inline void processInputLinks_GT(ap_uint<576> link_in[N_INPUT_LINKS_GT],
                                 GTInputVar EGspos[6],
                                 GTInputVar EGIsPos[6],
                                 GTInputVar JetsPos[6],
                                 GTInputVar TausPos[6],
                                 GTInputVar EGsNeg[6],
                                 GTInputVar EGIsNeg[6],
                                 GTInputVar JetsNeg[6],
                                 GTInputVar TausNeg[6],
                                 GCTsum Sums[2]) {
  for (int j = 0; j < 12; j++) {
    ap_uint<48> raw = link_in[0].range(j * 48 + 47, j * 48);
    if (j < 6)
      EGspos[j].unpack(raw, false);
    else
      EGIsPos[j - 6].unpack(raw, false);
  }

  for (int j = 0; j < 12; j++) {
    ap_uint<48> raw = link_in[1].range(j * 48 + 47, j * 48);
    if (j < 6)
      JetsPos[j].unpack(raw, true);
    else
      TausPos[j - 6].unpack(raw, true);
  }

  Sums[0].unpack(link_in[2]);

  for (int j = 0; j < 12; j++) {
    ap_uint<48> raw = link_in[3].range(j * 48 + 47, j * 48);
    if (j < 6)
      EGsNeg[j].unpack(raw, false);
    else
      EGIsNeg[j - 6].unpack(raw, false);
  }

  for (int j = 0; j < 12; j++) {
    ap_uint<48> raw = link_in[4].range(j * 48 + 47, j * 48);
    if (j < 6)
      JetsNeg[j].unpack(raw, true);
    else
      TausNeg[j - 6].unpack(raw, true);
  }

  Sums[1].unpack(link_in[5]);
}

inline void createOutputToGT(GTInputVar EGspos[6],
                             GTInputVar EGIsPos[6],
                             GTInputVar JetsPos[6],
                             GTInputVar TausPos[6],
                             GTInputVar EGsNeg[6],
                             GTInputVar EGIsNeg[6],
                             GTInputVar JetsNeg[6],
                             GTInputVar TausNeg[6],
                             GCTsum Sums[2],
                             GCTtoGT& combinedoutput) {
  combinedoutput.processSums(Sums[0], Sums[1]);
  combinedoutput.convertObjects(EGspos, EGsNeg, EGIsPos, EGIsNeg, JetsPos, JetsNeg, TausPos, TausNeg);
  combinedoutput.getcombinedGTfromIP();
  combinedoutput.putGTtoLink();
}

inline void algo_top_GT(ap_uint<576> link_in[N_INPUT_LINKS_GT], ap_uint<576> link_out[N_OUTPUT_LINKS_GT]) {
  GTInputVar EGspos[6], EGsNeg[6];
  GTInputVar EGIsPos[6], EGIsNeg[6];
  GTInputVar JetsPos[6], JetsNeg[6];
  GTInputVar TausPos[6], TausNeg[6];
  GCTsum Sums[2];

  GCTtoGT GCTtoGT;

  processInputLinks_GT(link_in, EGspos, EGIsPos, JetsPos, TausPos, EGsNeg, EGIsNeg, JetsNeg, TausNeg, Sums);
  createOutputToGT(EGspos, EGIsPos, JetsPos, TausPos, EGsNeg, EGIsNeg, JetsNeg, TausNeg, Sums, GCTtoGT);

  ap_uint<576> link_out_reg[N_OUTPUT_LINKS_GT] = {0};

  for (loop i = 0; i < N_OUTPUT_LINKS_GT; i++) {
    link_out_reg[i] = GCTtoGT.link[i];
  }

  processOutLinks_GT(link_out_reg, link_out);
}

}  // namespace p2gctsumGT

#endif
