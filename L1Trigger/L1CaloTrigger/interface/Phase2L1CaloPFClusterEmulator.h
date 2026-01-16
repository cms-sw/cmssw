#ifndef _PHASE_2_L1_CALO_PFCLUSTER_EMULATOR_H_
#define _PHASE_2_L1_CALO_PFCLUSTER_EMULATOR_H_

#include <cstdlib>

//    eta:  0  1  2  3  4   5  6  7  8   9 10 11 12  13 14 15 16  17 18 19 20
// 0             |                                                     |
// 1             |                                                     |
//               |-----------------------------------------------------|
// 2             |                                                     |
// 3             |                                                     |
// 4             |                                                     |
// 5             |                                                     |
//               | ----------------------------------------------------|
// 6             |                                                     |
// 7             |                                                     |
//
// 8 PFclusters are created in one 21x8 (2+17+2 x 2+4+2)

static constexpr int nTowerEta = 34;
static constexpr int nTowerPhi = 72;
static constexpr int nSLR = 36;
static constexpr int nTowerEtaSLR = 21;  // including overlap: 2+17+2
static constexpr int nTowerPhiSLR = 8;   // including overlap: 2+4+2
static constexpr int nPFClusterSLR = 8;
static constexpr int nHfEta = 24;
static constexpr int nHfPhi = 72;
static constexpr int nHfRegions = 12;

namespace gctpf {

  typedef struct {
    float et;
    int eta;
    int phi;
  } GCTpfcluster_t;

  typedef struct {
    GCTpfcluster_t GCTpfclusters[nPFClusterSLR];
  } PFcluster_t;

  typedef struct {
    float et;
    int eta;
    int phi;
  } GCTint_t;

  typedef struct {
    GCTint_t t[nTowerPhiSLR];
  } GCTEtaStrip_t;

  typedef struct {
    GCTint_t t[nHfPhi / 6];
  } GCTEtaHFStrip_t;

  typedef struct {
    GCTint_t p[nTowerEtaSLR - 2];
  } GCTEtaStripPeak_t;

  typedef struct {
    GCTint_t p[nHfEta];
  } GCTEtaHFStripPeak_t;

  typedef struct {
    GCTEtaStrip_t s[nTowerEtaSLR];
  } Region_t;

  typedef struct {
    GCTEtaHFStrip_t s[nHfEta];
  } RegionHF_t;

  inline GCTint_t bestOf2(const GCTint_t& t0, const GCTint_t& t1) {
    GCTint_t x;
    x = (t0.et > t1.et) ? t0 : t1;

    return x;
  }

  inline GCTint_t getPeakOfStrip(const GCTEtaStrip_t& etaStrip) {
    GCTint_t best12 = bestOf2(etaStrip.t[1], etaStrip.t[2]);
    GCTint_t best34 = bestOf2(etaStrip.t[3], etaStrip.t[4]);
    GCTint_t best56 = bestOf2(etaStrip.t[5], etaStrip.t[6]);
    GCTint_t best1234 = bestOf2(best12, best34);
    GCTint_t bestAll = bestOf2(best1234, best56);

    return bestAll;
  }

  inline GCTint_t getPeakOfHFStrip(const GCTEtaHFStrip_t& etaStrip) {
    GCTint_t best01 = bestOf2(etaStrip.t[0], etaStrip.t[1]);
    GCTint_t best23 = bestOf2(etaStrip.t[2], etaStrip.t[3]);
    GCTint_t best45 = bestOf2(etaStrip.t[4], etaStrip.t[5]);
    GCTint_t best67 = bestOf2(etaStrip.t[6], etaStrip.t[7]);
    GCTint_t best89 = bestOf2(etaStrip.t[8], etaStrip.t[9]);
    GCTint_t best1011 = bestOf2(etaStrip.t[10], etaStrip.t[11]);
    GCTint_t best0123 = bestOf2(best01, best23);
    GCTint_t best4567 = bestOf2(best45, best67);
    GCTint_t best891011 = bestOf2(best89, best1011);
    GCTint_t best01234567 = bestOf2(best0123, best4567);
    GCTint_t bestAll = bestOf2(best01234567, best891011);

    return bestAll;
  }

  inline GCTint_t getPeakBin(const GCTEtaStripPeak_t& etaStripPeak) {
    GCTint_t best01 = bestOf2(etaStripPeak.p[0], etaStripPeak.p[1]);
    GCTint_t best23 = bestOf2(etaStripPeak.p[2], etaStripPeak.p[3]);
    GCTint_t best45 = bestOf2(etaStripPeak.p[4], etaStripPeak.p[5]);
    GCTint_t best67 = bestOf2(etaStripPeak.p[6], etaStripPeak.p[7]);
    GCTint_t best89 = bestOf2(etaStripPeak.p[8], etaStripPeak.p[9]);
    GCTint_t best1011 = bestOf2(etaStripPeak.p[10], etaStripPeak.p[11]);
    GCTint_t best1213 = bestOf2(etaStripPeak.p[12], etaStripPeak.p[13]);
    GCTint_t best1415 = bestOf2(etaStripPeak.p[14], etaStripPeak.p[15]);
    GCTint_t best1617 = bestOf2(etaStripPeak.p[16], etaStripPeak.p[17]);
    GCTint_t best0123 = bestOf2(best01, best23);
    GCTint_t best4567 = bestOf2(best45, best67);
    GCTint_t best891011 = bestOf2(best89, best1011);
    GCTint_t best12131415 = bestOf2(best1213, best1415);
    GCTint_t best01234567 = bestOf2(best0123, best4567);
    GCTint_t best01234567891011 = bestOf2(best01234567, best891011);
    GCTint_t best121314151617 = bestOf2(best12131415, best1617);
    GCTint_t best12131415161718 = bestOf2(best121314151617, etaStripPeak.p[18]);
    GCTint_t bestAll = bestOf2(best01234567891011, best12131415161718);

    return bestAll;
  }

  inline GCTint_t getPeakBinHF(const GCTEtaHFStripPeak_t& etaStripPeak) {
    GCTint_t best01 = bestOf2(etaStripPeak.p[0], etaStripPeak.p[1]);
    GCTint_t best23 = bestOf2(etaStripPeak.p[2], etaStripPeak.p[3]);
    GCTint_t best45 = bestOf2(etaStripPeak.p[4], etaStripPeak.p[5]);
    GCTint_t best67 = bestOf2(etaStripPeak.p[6], etaStripPeak.p[7]);
    GCTint_t best89 = bestOf2(etaStripPeak.p[8], etaStripPeak.p[9]);
    GCTint_t best1011 = bestOf2(etaStripPeak.p[10], etaStripPeak.p[11]);
    GCTint_t best1213 = bestOf2(etaStripPeak.p[12], etaStripPeak.p[13]);
    GCTint_t best1415 = bestOf2(etaStripPeak.p[14], etaStripPeak.p[15]);
    GCTint_t best1617 = bestOf2(etaStripPeak.p[16], etaStripPeak.p[17]);
    GCTint_t best1819 = bestOf2(etaStripPeak.p[18], etaStripPeak.p[19]);
    GCTint_t best2021 = bestOf2(etaStripPeak.p[20], etaStripPeak.p[21]);
    GCTint_t best2223 = bestOf2(etaStripPeak.p[22], etaStripPeak.p[23]);
    GCTint_t best0123 = bestOf2(best01, best23);
    GCTint_t best4567 = bestOf2(best45, best67);
    GCTint_t best891011 = bestOf2(best89, best1011);
    GCTint_t best12131415 = bestOf2(best1213, best1415);
    GCTint_t best16171819 = bestOf2(best1617, best1819);
    GCTint_t best20212223 = bestOf2(best2021, best2223);
    GCTint_t best0to7 = bestOf2(best0123, best4567);
    GCTint_t best8to15 = bestOf2(best891011, best12131415);
    GCTint_t best16to23 = bestOf2(best16171819, best20212223);
    GCTint_t best0to15 = bestOf2(best0to7, best8to15);
    GCTint_t bestAll = bestOf2(best0to15, best16to23);

    return bestAll;
  }

  inline GCTint_t getPeakPosition(const Region_t& region) {
    GCTEtaStripPeak_t etaPeak;
    for (int i = 0; i < nTowerEtaSLR - 2; i++) {
      etaPeak.p[i] = getPeakOfStrip(region.s[i + 1]);
    }
    GCTint_t max = getPeakBin(etaPeak);

    return max;
  }

  inline GCTint_t getPeakPositionHF(const RegionHF_t& region) {
    GCTEtaHFStripPeak_t etaPeak;
    for (int i = 0; i < nHfEta; i++) {
      etaPeak.p[i] = getPeakOfHFStrip(region.s[i]);
    }
    GCTint_t max = getPeakBinHF(etaPeak);

    return max;
  }

  inline Region_t initStructure(float temp[nTowerEtaSLR][nTowerPhiSLR]) {
    Region_t r;

    for (int i = 0; i < nTowerPhiSLR; i++) {
      for (int j = 0; j < nTowerEtaSLR; j++) {
        r.s[j].t[i].et = temp[j][i];
        r.s[j].t[i].eta = j;
        r.s[j].t[i].phi = i;
      }
    }

    return r;
  }

  inline RegionHF_t initStructureHF(float temp[nHfEta][nHfPhi / 6]) {
    RegionHF_t r;

    for (int i = 0; i < nHfPhi / 6; i++) {
      for (int j = 0; j < nHfEta; j++) {
        r.s[j].t[i].et = temp[j][i];
        r.s[j].t[i].eta = j;
        r.s[j].t[i].phi = i;
      }
    }

    return r;
  }

  inline float getEt(float temp[nTowerEtaSLR][nTowerPhiSLR], int eta, int phi) {
    float et_sumEta[3] = {0., 0., 0.};

    for (int i = 0; i < (nTowerEtaSLR - 2); i++) {
      for (int j = 0; j < (nTowerPhiSLR - 2); j++) {
        if (i + 1 == eta && j + 1 == phi) {
          for (int k = 0; k < 3; k++) {
            et_sumEta[k] = temp[i + k][j] + temp[i + k][j + 1] + temp[i + k][j + 2];
          }
        }
      }
    }

    float pfcluster_et = et_sumEta[0] + et_sumEta[1] + et_sumEta[2];

    return pfcluster_et;
  }

  inline float getEtHF(float temp[nHfEta][nHfPhi / 6], int eta, int phi) {
    float tempX[nHfEta + 2][nHfPhi / 6 + 2];
    float et_sumEta[3] = {0., 0., 0.};

    for (int i = 0; i < nHfEta + 2; i++) {
      for (int k = 0; k < nHfPhi / 6 + 2; k++) {
        tempX[i][k] = 0;
      }
    }

    for (int i = 0; i < nHfEta; i++) {
      for (int k = 0; k < nHfPhi / 6; k++) {
        tempX[i + 1][k + 1] = temp[i][k];
      }
    }

    for (int i = 0; i < nHfEta; i++) {
      for (int j = 0; j < nHfPhi / 6; j++) {
        if (i == eta && j == phi) {
          for (int k = 0; k < 3; k++) {
            et_sumEta[k] = tempX[i + k][j] + tempX[i + k][j + 1] + tempX[i + k][j + 2];
          }
        }
      }
    }

    float pfcluster_et = et_sumEta[0] + et_sumEta[1] + et_sumEta[2];

    return pfcluster_et;
  }

  inline void RemoveTmp(float temp[nTowerEtaSLR][nTowerPhiSLR], int eta, int phi) {
    for (int i = 0; i < nTowerEtaSLR; i++) {
      if (i + 1 >= eta && i <= eta + 1) {
        for (int j = 0; j < nTowerPhiSLR; j++) {
          if (j + 1 >= phi && j <= phi + 1) {
            temp[i][j] = 0;
          }
        }
      }
    }

    return;
  }

  inline void RemoveTmpHF(float temp[nHfEta][nHfPhi / 6], int eta, int phi) {
    for (int i = 0; i < nHfEta; i++) {
      if (i + 1 >= eta && i <= eta + 1) {
        for (int j = 0; j < nHfPhi / 6; j++) {
          if (j + 1 >= phi && j <= phi + 1) {
            temp[i][j] = 0;
          }
        }
      }
    }

    return;
  }

  inline GCTpfcluster_t recoPfcluster(float temporary[nTowerEtaSLR][nTowerPhiSLR], int etaoffset, int phioffset) {
    GCTpfcluster_t pfclusterReturn;

    Region_t region;

    region = initStructure(temporary);

    GCTint_t regionMax = getPeakPosition(region);

    float pfcluster_et = getEt(temporary, regionMax.eta, regionMax.phi);
    float pfcluster_eta = regionMax.eta - 2 + etaoffset;
    float pfcluster_phi = regionMax.phi - 2 + phioffset;

    RemoveTmp(temporary, regionMax.eta, regionMax.phi);

    if (!(regionMax.eta >= 2 && regionMax.eta < (nTowerEtaSLR - 2) && regionMax.phi >= 2 &&
          regionMax.phi < (nTowerPhiSLR - 2))) {
      pfcluster_et = 0;   // set energy to be zero if maximum energy tower is not within the unique region
      pfcluster_eta = 2;  // choose the default to be at one corner of the unique region
      pfcluster_phi = 2;  // choose the default to be at one corner of the unique region
    }

    pfclusterReturn.et = pfcluster_et;
    pfclusterReturn.eta = pfcluster_eta;
    pfclusterReturn.phi = pfcluster_phi;

    return pfclusterReturn;
  }

  inline GCTpfcluster_t recoPfclusterHF(float temporary[nHfEta][nHfPhi / 6], int etaoffset, int phioffset) {
    GCTpfcluster_t pfclusterReturn;

    RegionHF_t region;

    region = initStructureHF(temporary);

    GCTint_t regionMax = getPeakPositionHF(region);

    float pfcluster_et = getEtHF(temporary, regionMax.eta, regionMax.phi);

    RemoveTmpHF(temporary, regionMax.eta, regionMax.phi);

    pfclusterReturn.et = pfcluster_et;
    pfclusterReturn.eta = regionMax.eta + etaoffset;
    pfclusterReturn.phi = regionMax.phi + phioffset;
    if (pfclusterReturn.phi < 0)
      pfclusterReturn.phi += nHfPhi;

    return pfclusterReturn;
  }

  inline PFcluster_t pfcluster(float temporary[nTowerEtaSLR][nTowerPhiSLR], int etaoffset, int phioffset) {
    GCTpfcluster_t pfcluster[nPFClusterSLR];

    for (int i = 0; i < nPFClusterSLR; i++) {
      pfcluster[i] = recoPfcluster(temporary, etaoffset, phioffset);
    }

    PFcluster_t GCTPfclusters;

    for (int i = 0; i < nPFClusterSLR; i++) {
      GCTPfclusters.GCTpfclusters[i].et = pfcluster[i].et;
      GCTPfclusters.GCTpfclusters[i].eta = pfcluster[i].eta;
      GCTPfclusters.GCTpfclusters[i].phi = pfcluster[i].phi;
    }

    return GCTPfclusters;
  }

  inline PFcluster_t pfclusterHF(float temporary[nHfEta][nHfPhi / 6], int etaoffset, int phioffset) {
    GCTpfcluster_t pfcluster[nPFClusterSLR];

    for (int i = 0; i < nPFClusterSLR; i++) {
      pfcluster[i] = recoPfclusterHF(temporary, etaoffset, phioffset);
    }

    PFcluster_t GCTPfclusters;

    for (int i = 0; i < nPFClusterSLR; i++) {
      GCTPfclusters.GCTpfclusters[i].et = pfcluster[i].et;
      GCTPfclusters.GCTpfclusters[i].eta = pfcluster[i].eta;
      GCTPfclusters.GCTpfclusters[i].phi = pfcluster[i].phi;
    }

    return GCTPfclusters;
  }

}  // namespace gctpf

#endif
