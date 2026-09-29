//------------------------------------
// Header and data formats for GCT SumCard to GT emulator
// (based heavily on algo_top.h in TO_GT_IP firmware repo)
//------------------------------------
#ifndef L1Trigger_L1CaloTrigger_GCTSumToGT_h
#define L1Trigger_L1CaloTrigger_GCTSumToGT_h

#include <ap_int.h>
#include <algorithm>
#include <cstdint>
#include <iostream>

#include "L1Trigger/L1CaloTrigger/interface/GCTSum_h.h"

namespace p2gctsumGT {

static constexpr int N_INPUT_LINKS_GT = 6;
static constexpr int N_OUTPUT_LINKS_GT = 6;

using p2gctsum::GCTsum;
typedef ap_uint<10> loop;

inline ap_uint<64> pack_scalar16_word(ap_uint<16> value) {
  ap_uint<64> out = 0;
  out[0] = 1;
  out.range(16, 1) = value;
  return out;
}

class GTInputVar {
public:
  ap_uint<12> ET;
  ap_uint<10> Eta;
  ap_uint<9> Phi;
  ap_uint<4> PtClusterSeed;
  ap_uint<1> isBarrel;

  GTInputVar() : ET(0), Eta(0), Phi(0), PtClusterSeed(0), isBarrel(0) {}

  void unpack(ap_uint<48> input, bool hasSeed) {
    ET = input.range(11, 0);
    isBarrel = input.range(47, 47);
    if (hasSeed) {
      Eta = input.range(17, 12);
      Phi = input.range(26, 18);
      PtClusterSeed = input.range(30, 27);
    } else {
      Eta = input.range(18, 12);
      Eta[7] = input[28];
      Phi = input.range(27, 19);
      PtClusterSeed = 0;
    }
  }
};

class GTvar {
public:
  ap_uint<1> isValid;
  ap_uint<16> ET;
  ap_uint<13> Phi;
  ap_uint<14> Eta;
  ap_uint<4> PtClusterSeed;
  ap_uint<20> Spare;

  GTvar() {
    isValid = 0;
    ET = 0;
    Phi = 0;
    Eta = 0;
    PtClusterSeed = 0;
    Spare = 0;
  }

  void getGTvar(ap_uint<64> i) {
    this->isValid = i.range(0, 0);
    this->ET = i.range(16, 1);
    this->Phi = i.range(29, 17);
    this->Eta = i.range(43, 30);
    this->Spare = i.range(63, 44);
  }

  ap_uint<64> pack() const {
    return (ap_uint<64>)isValid | ((ap_uint<64>)ET << 1) | ((ap_uint<64>)Phi << 17) | ((ap_uint<64>)Eta << 30) |
           ((ap_uint<64>)Spare << 44);
  }

  void convertAndPack(GTInputVar& in) {
    if (in.ET == 0) {
      isValid = 0;
      ET = 0;
      Phi = 0;
      Eta = 0;
      PtClusterSeed = 0;
      Spare = 0;
      return;
    }

    this->isValid = 1;
    this->ET = (ap_uint<16>)in.ET << 4;

    ap_uint<13> PhiL = (ap_uint<13>)in.Phi;
    if (in.Phi >= 179 && in.Phi <= 182) {
      PhiL = 178;
    } else if (in.Phi >= 183) {
      PhiL = (360 - in.Phi) | 0x1000;
    }
    this->Phi = (ap_uint<13>(PhiL & 0x1000)) | (ap_uint<13>(PhiL & 0x1FF) * 23);

    this->Eta = (ap_uint<14>(in.Eta & 0x200)) << 4 | (ap_uint<14>(in.Eta & 0x1FF) * 23);

    this->PtClusterSeed = in.PtClusterSeed;
    this->Spare = (ap_uint<20>)in.PtClusterSeed;
  }
};

class GCTtoGT {
public:
  GTvar EGspos[6];
  GTvar EGsneg[6];
  GTvar EGIspos[6];
  GTvar EGIsneg[6];
  GTvar Jetspos[6];
  GTvar Jetsneg[6];
  GTvar Tauspos[6];
  GTvar Tausneg[6];

  ap_uint<64> Sums[4];
  ap_uint<17> SumETTotal;
  ap_uint<576> linkOutput[6];
  ap_uint<576> link[6];

  GCTtoGT() {
    for (int i = 0; i < 4; i++)
      Sums[i] = 0;
    SumETTotal = 0;
    for (int i = 0; i < 6; i++) {
      linkOutput[i] = 0;
      link[i] = 0;
    }
  }

  void convertObjects(GTInputVar _EGspos[6],
                      GTInputVar _EGsneg[6],
                      GTInputVar _EGIspos[6],
                      GTInputVar _EGIsneg[6],
                      GTInputVar _Jetspos[6],
                      GTInputVar _Jetsneg[6],
                      GTInputVar _Tauspos[6],
                      GTInputVar _Tausneg[6]) {
    for (int i = 0; i < 6; i++) {
      EGspos[i].convertAndPack(_EGspos[i]);
      EGsneg[i].convertAndPack(_EGsneg[i]);
      EGIspos[i].convertAndPack(_EGIspos[i]);
      EGIsneg[i].convertAndPack(_EGIsneg[i]);
      Jetspos[i].convertAndPack(_Jetspos[i]);
      Jetsneg[i].convertAndPack(_Jetsneg[i]);
      Tauspos[i].convertAndPack(_Tauspos[i]);
      Tausneg[i].convertAndPack(_Tausneg[i]);
    }
  }

  void processSums(const GCTsum& sumsPos, const GCTsum& sumsNeg) {
    const ap_int<17> totalEx = (ap_int<17>)sumsPos.Ex + (ap_int<17>)sumsNeg.Ex;
    const ap_int<17> totalEy = (ap_int<17>)sumsPos.Ey + (ap_int<17>)sumsNeg.Ey;
    const ap_int<17> totalHtx = (ap_int<17>)sumsPos.Htx + (ap_int<17>)sumsNeg.Htx;
    const ap_int<17> totalHty = (ap_int<17>)sumsPos.Hty + (ap_int<17>)sumsNeg.Hty;
    const ap_uint<17> totalHt = (ap_uint<17>)sumsPos.Ht + (ap_uint<17>)sumsNeg.Ht;
    const ap_uint<17> nObjTotal = (ap_uint<17>)sumsPos.NObj + (ap_uint<17>)sumsNeg.NObj;
    this->SumETTotal = (ap_uint<17>)sumsPos.SumET + (ap_uint<17>)sumsNeg.SumET;
    this->Sums[0] = p2gctsum::gcts::packVectorSum(totalEx, totalEy, SumETTotal);
    this->Sums[1] = p2gctsum::gcts::packVectorSum(totalHtx, totalHty, totalHt);
    this->Sums[2] = pack_scalar16_word(p2gctsum::gcts::saturate16((ap_uint<24>)nObjTotal));
    this->Sums[3] = 0;
  }

  void getcombinedGTfromIP() {
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 9; j++) {
        ap_uint<10> start = 64 * j;
        ap_uint<10> end = start + 63;
        ap_uint<64> data = 0;

        switch (i) {
          case 0:
            if (j < 6)
              data = EGspos[j].pack();
            else
              data = EGsneg[j - 6].pack();
            break;
          case 1:
            if (j < 3)
              data = EGsneg[j + 3].pack();
            else
              data = EGIspos[j - 3].pack();
            break;
          case 2:
            if (j < 6)
              data = EGIsneg[j].pack();
            else
              data = Jetspos[j - 6].pack();
            break;
          case 3:
            if (j < 3)
              data = Jetspos[j + 3].pack();
            else
              data = Jetsneg[j - 3].pack();
            break;
          case 4:
            if (j < 6)
              data = Tauspos[j].pack();
            else
              data = Tausneg[j - 6].pack();
            break;
          case 5:
            if (j < 3)
              data = Tausneg[j + 3].pack();
            else if (j < 7)
              data = this->Sums[j - 3];
            else
              data = 0;
            break;
        }

        linkOutput[i].range(end, start) = data;
      }
    }
  }

  void putGTtoLink() {
    for (loop i = 0; i < 6; i++) {
      link[i] = linkOutput[i];
    }
  }

  GCTtoGT& operator=(const GCTtoGT& rhs) {
    for (int i = 0; i < 6; i++) {
      this->EGspos[i] = rhs.EGspos[i];
      this->EGsneg[i] = rhs.EGsneg[i];
      this->EGIspos[i] = rhs.EGIspos[i];
      this->EGIsneg[i] = rhs.EGIsneg[i];
      this->Jetspos[i] = rhs.Jetspos[i];
      this->Jetsneg[i] = rhs.Jetsneg[i];
      this->Tauspos[i] = rhs.Tauspos[i];
      this->Tausneg[i] = rhs.Tausneg[i];
      this->linkOutput[i] = rhs.linkOutput[i];
      this->link[i] = rhs.link[i];
    }
    for (int i = 0; i < 4; ++i) {
      this->Sums[i] = rhs.Sums[i];
    }
    this->SumETTotal = rhs.SumETTotal;
    return *this;
  }
};

void algo_top_GT(ap_uint<576> link_in[N_INPUT_LINKS_GT], ap_uint<576> link_out[N_OUTPUT_LINKS_GT]);

}  // namespace p2gctsumGT

#endif
