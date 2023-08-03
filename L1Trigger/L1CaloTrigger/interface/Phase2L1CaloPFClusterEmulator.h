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

namespace gctpf {

  typedef struct {
    float et;
    int eta;
    int phi;
  } GCTpfcluster_t;

  typedef struct {
    GCTpfcluster_t GCTpfclusters[nPFClusterSLR];
  } GCTPfcluster_t;

  typedef struct {
    float et;
    int eta;
    int phi;
  } GCTint_t;

  typedef struct {
    GCTint_t t[nTowerPhiSLR];
  } gctEtaStrip_t;

  typedef struct {
    GCTint_t p[nTowerEtaSLR - 2];
  } gctEtaStripPeak_t;

  typedef struct {
    gctEtaStrip_t s[nTowerEtaSLR];
  } region_t;

  inline GCTint_t bestOf2(const GCTint_t& t0, const GCTint_t& t1) {
    GCTint_t x;
    x = (t0.et > t1.et) ? t0 : t1;

    return x;
  }

  inline GCTint_t getPeakOfStrip(const gctEtaStrip_t& etaStrip) {
    GCTint_t best12 = bestOf2(etaStrip.t[1], etaStrip.t[2]);
    GCTint_t best34 = bestOf2(etaStrip.t[3], etaStrip.t[4]);
    GCTint_t best56 = bestOf2(etaStrip.t[5], etaStrip.t[6]);
    GCTint_t best1234 = bestOf2(best12, best34);
    GCTint_t bestAll = bestOf2(best1234, best56);

    return bestAll;
  }

  inline GCTint_t getPeakBin(const gctEtaStripPeak_t& etaStripPeak) {
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

  inline GCTint_t getPeakPosition(const region_t& region) {
    gctEtaStripPeak_t etaPeak;
    for (int i = 0; i < 19; i++) {
      etaPeak.p[i] = getPeakOfStrip(region.s[i + 1]);
    }
    GCTint_t max = getPeakBin(etaPeak);

    return max;
  }

  inline region_t initStructure(float temp[nTowerEtaSLR][nTowerPhiSLR]) {
    region_t r;

    for (int i = 0; i < nTowerPhiSLR; i++) {
      for (int j = 0; j < nTowerEtaSLR; j++) {
        r.s[j].t[i].et = temp[j][i];
        r.s[j].t[i].eta = j;
        r.s[j].t[i].phi = i;
      }
    }

    return r;
  }

  inline float getEt(float temp[nTowerEtaSLR][nTowerPhiSLR], int eta, int phi) {
    float et_sumEta[3];

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

  inline GCTpfcluster_t recoPfcluster(float temporary[nTowerEtaSLR][nTowerPhiSLR], int etaoffset, int phioffset) {
    GCTpfcluster_t pfclusterReturn;

    region_t region;

    region = initStructure(temporary);

    GCTint_t regionMax = getPeakPosition(region);

    float pfcluster_et = getEt(temporary, regionMax.eta, regionMax.phi);

    RemoveTmp(temporary, regionMax.eta, regionMax.phi);

    if (!(regionMax.eta >= 2 && regionMax.eta < (nTowerEtaSLR - 2) && regionMax.phi >= 2 &&
          regionMax.phi < (nTowerPhiSLR - 2)))
      pfcluster_et = 0;

    pfclusterReturn.et = pfcluster_et;
    pfclusterReturn.eta = regionMax.eta - 2 + etaoffset;
    pfclusterReturn.phi = regionMax.phi - 2 + phioffset;

    return pfclusterReturn;
  }

  inline GCTPfcluster_t pfcluster(float temporary[nTowerEtaSLR][nTowerPhiSLR], int etaoffset, int phioffset) {
    GCTpfcluster_t pfcluster[nPFClusterSLR];

    for (int i = 0; i < nPFClusterSLR; i++) {
      pfcluster[i] = recoPfcluster(temporary, etaoffset, phioffset);
    }

    GCTPfcluster_t GCTPfclusters;

    for (int i = 0; i < nPFClusterSLR; i++) {
      GCTPfclusters.GCTpfclusters[i].et = pfcluster[i].et;
      GCTPfclusters.GCTpfclusters[i].eta = pfcluster[i].eta;
      GCTPfclusters.GCTpfclusters[i].phi = pfcluster[i].phi;
    }

    return GCTPfclusters;
  }

}  // namespace gctpf

#endif
