#ifndef L1TRIGGER_L1CALOTRIGGER_PHASE2L1CALOJETEMULATOR_H
#define L1TRIGGER_L1CALOTRIGGER_PHASE2L1CALOJETEMULATOR_H

#include "DataFormats/L1TCalorimeterPhase2/interface/Phase2L1CaloJet.h"
#include <cstdlib>

static constexpr int nBarrelEta = 34;
static constexpr int nBarrelPhi = 72;
static constexpr int nHgcalEta = 36;
static constexpr int nHgcalPhi = 72;
static constexpr int nHfEta = 24;
static constexpr int nHfPhi = 72;
static constexpr int nSTEta = 6;
static constexpr int nSTEta_hf = 4;
static constexpr int nSTPhi = 24;
static constexpr int nJets = 10;

namespace gctobj {

  class towerMax {
  public:
    float energy;
    int phi;
    int eta;
    float energyMax;
    int phiMax;
    int etaMax;
    int phiCenter;
    int etaCenter;

    towerMax() {
      energy = 0;
      phi = 0;
      eta = 0;
      energyMax = 0;
      phiMax = 0;
      etaMax = 0;
      phiCenter = 0;
      etaCenter = 0;
    }

    towerMax& operator=(const towerMax& rhs) {
      energy = rhs.energy;
      phi = rhs.phi;
      eta = rhs.eta;
      energyMax = rhs.energyMax;
      phiMax = rhs.phiMax;
      etaMax = rhs.etaMax;
      phiCenter = rhs.phiCenter;
      etaCenter = rhs.etaCenter;
      return *this;
    }
  };

  class jetInfo {
  public:
    float seedEnergy;
    float energy;
    float tauEt;
    int phi;
    int eta;
    float energyMax;
    int phiMax;
    int etaMax;
    int phiCenter;
    int etaCenter;

    jetInfo() {
      seedEnergy = 0;
      energy = 0;
      tauEt = 0;
      phi = 0;
      eta = 0;
      energyMax = 0;
      phiMax = 0;
      etaMax = 0;
      phiCenter = 0;
      etaCenter = 0;
    }

    jetInfo& operator=(const jetInfo& rhs) {
      seedEnergy = rhs.seedEnergy;
      energy = rhs.energy;
      tauEt = rhs.tauEt;
      phi = rhs.phi;
      eta = rhs.eta;
      energyMax = rhs.energyMax;
      phiMax = rhs.phiMax;
      etaMax = rhs.etaMax;
      phiCenter = rhs.phiCenter;
      etaCenter = rhs.etaCenter;
      return *this;
    }
  };

  typedef struct {
    float et;
    int eta;
    int phi;
    float towerEt;
    int towerEta;
    int towerPhi;
    int centerEta;
    int centerPhi;
  } GCTsupertower_t;

  typedef struct {
    GCTsupertower_t cr[nSTPhi];
  } etaStrip_t;

  typedef struct {
    GCTsupertower_t etaStrip[nSTEta];
  } hgcalRegion_t;

  typedef struct {
    GCTsupertower_t pk[nSTEta];
  } etaStripPeak_t;

  typedef struct {
    float et;
    float hoe;
  } GCTtower_t;

  typedef struct {
    GCTtower_t GCTtower[nBarrelEta / 2][nBarrelPhi];
  } GCTintTowers_t;

  inline int getPeakBinOf3(float et0, float et1, float et2) {
    int x;
    float temp;
    if (et0 > et1) {
      x = 0;
      temp = et0;
    } else {
      x = 1;
      temp = et1;
    }
    if (et2 > temp) {
      x = 2;
    }
    return x;
  }

  inline int getEtCenterOf3(float et0, float et1, float et2) {
    float etSum = et0 + et1 + et2;
    float iEtSum = 0.5 * et0 + 1.5 * et1 + 2.5 * et2;
    int iAve = 0xEEF;
    if (iEtSum <= etSum)
      iAve = 0;
    else if (iEtSum <= 2 * etSum)
      iAve = 1;
    else
      iAve = 2;
    return iAve;
  }

  inline void makeST(const float GCTintTowers[nBarrelEta / 2][nBarrelPhi],
                     GCTsupertower_t supertower_return[nSTEta][nSTPhi]) {
    float et_sumEta[nSTEta][nSTPhi][3];
    float stripEta[nSTEta][nSTPhi][3];
    float stripPhi[nSTEta][nSTPhi][3];

    float ex_et[nBarrelEta / 2 + 1][nBarrelPhi];
    for (int j = 0; j < nBarrelPhi; j++) {
      ex_et[nBarrelEta / 2][j] = 0;
      for (int i = 0; i < nBarrelEta / 2; i++) {
        ex_et[i][j] = GCTintTowers[i][j];
      }
    }

    int index_i = 0;
    int index_j = 0;
    for (int i = 0; i < nBarrelEta / 2 + 1; i += 3) {  // 17+1 to divide into 6 super towers
      index_j = 0;
      for (int j = 0; j < nBarrelPhi; j += 3) {  // 72 phi to 24 super towers
        stripEta[index_i][index_j][0] = ex_et[i][j] + ex_et[i][j + 1] + ex_et[i][j + 2];
        stripEta[index_i][index_j][1] = ex_et[i + 1][j] + ex_et[i + 1][j + 1] + ex_et[i + 1][j + 2];
        stripEta[index_i][index_j][2] = ex_et[i + 2][j] + ex_et[i + 2][j + 1] + ex_et[i + 2][j + 2];
        stripPhi[index_i][index_j][0] = ex_et[i][j] + ex_et[i + 1][j] + ex_et[i + 2][j];
        stripPhi[index_i][index_j][1] = ex_et[i][j + 1] + ex_et[i + 1][j + 1] + ex_et[i + 2][j + 1];
        stripPhi[index_i][index_j][2] = ex_et[i][j + 2] + ex_et[i + 1][j + 2] + ex_et[i + 2][j + 2];
        for (int k = 0; k < 3; k++) {
          et_sumEta[index_i][index_j][k] = ex_et[i + k][j] + ex_et[i + k][j + 1] + ex_et[i + k][j + 2];
        }
        index_j++;
      }
      index_i++;
    }
    for (int i = 0; i < nSTEta; i++) {
      for (int j = 0; j < nSTPhi; j++) {
        GCTsupertower_t temp;
        float supertowerEt = et_sumEta[i][j][0] + et_sumEta[i][j][1] + et_sumEta[i][j][2];
        temp.et = supertowerEt;
        temp.eta = i;
        temp.phi = j;
        int peakEta = getPeakBinOf3(stripEta[i][j][0], stripEta[i][j][1], stripEta[i][j][2]);
        temp.towerEta = peakEta;
        int peakPhi = getPeakBinOf3(stripPhi[i][j][0], stripPhi[i][j][1], stripPhi[i][j][2]);
        temp.towerPhi = peakPhi;
        float peakEt = ex_et[i * 3 + peakEta][j * 3 + peakPhi];
        temp.towerEt = peakEt;
        int cEta = getEtCenterOf3(stripEta[i][j][0], stripEta[i][j][1], stripEta[i][j][2]);
        temp.centerEta = cEta;
        int cPhi = getEtCenterOf3(stripPhi[i][j][0], stripPhi[i][j][1], stripPhi[i][j][2]);
        temp.centerPhi = cPhi;
        supertower_return[i][j] = temp;
      }
    }
  }

  inline void makeST_hgcal(const float hgcalTowers[nHgcalEta / 2][nHgcalPhi],
                           GCTsupertower_t supertower_return[nSTEta][nSTPhi]) {
    float et_sumEta[nSTEta][nSTPhi][3];
    float stripEta[nSTEta][nSTPhi][3];
    float stripPhi[nSTEta][nSTPhi][3];

    int index_i = 0;
    int index_j = 0;
    for (int i = 0; i < nHgcalEta / 2; i += 3) {  // 18 eta to 6 super towers
      index_j = 0;
      for (int j = 0; j < nHgcalPhi; j += 3) {  // 72 phi to 24 super towers
        stripEta[index_i][index_j][0] = hgcalTowers[i][j] + hgcalTowers[i][j + 1] + hgcalTowers[i][j + 2];
        stripEta[index_i][index_j][1] = hgcalTowers[i + 1][j] + hgcalTowers[i + 1][j + 1] + hgcalTowers[i + 1][j + 2];
        stripEta[index_i][index_j][2] = hgcalTowers[i + 2][j] + hgcalTowers[i + 2][j + 1] + hgcalTowers[i + 2][j + 2];
        stripPhi[index_i][index_j][0] = hgcalTowers[i][j] + hgcalTowers[i + 1][j] + hgcalTowers[i + 2][j];
        stripPhi[index_i][index_j][1] = hgcalTowers[i][j + 1] + hgcalTowers[i + 1][j + 1] + hgcalTowers[i + 2][j + 1];
        stripPhi[index_i][index_j][2] = hgcalTowers[i][j + 2] + hgcalTowers[i + 1][j + 2] + hgcalTowers[i + 2][j + 2];
        for (int k = 0; k < 3; k++) {
          et_sumEta[index_i][index_j][k] =
              hgcalTowers[i + k][j] + hgcalTowers[i + k][j + 1] + hgcalTowers[i + k][j + 2];
        }
        index_j++;
      }
      index_i++;
    }

    for (int i = 0; i < nSTEta; i++) {
      for (int j = 0; j < nSTPhi; j++) {
        GCTsupertower_t temp;
        temp.et = 0;
        temp.eta = 0;
        temp.phi = 0;
        temp.towerEta = 0;
        temp.towerPhi = 0;
        temp.towerEt = 0;
        temp.centerEta = 0;
        temp.centerPhi = 0;
        float supertowerEt = et_sumEta[i][j][0] + et_sumEta[i][j][1] + et_sumEta[i][j][2];
        temp.et = supertowerEt;
        temp.eta = i;
        temp.phi = j;
        int peakEta = getPeakBinOf3(stripEta[i][j][0], stripEta[i][j][1], stripEta[i][j][2]);
        temp.towerEta = peakEta;
        int peakPhi = getPeakBinOf3(stripPhi[i][j][0], stripPhi[i][j][1], stripPhi[i][j][2]);
        temp.towerPhi = peakPhi;
        float peakEt = hgcalTowers[i * 3 + peakEta][j * 3 + peakPhi];
        temp.towerEt = peakEt;
        int cEta = getEtCenterOf3(stripEta[i][j][0], stripEta[i][j][1], stripEta[i][j][2]);
        temp.centerEta = cEta;
        int cPhi = getEtCenterOf3(stripPhi[i][j][0], stripPhi[i][j][1], stripPhi[i][j][2]);
        temp.centerPhi = cPhi;
        supertower_return[i][j] = temp;
      }
    }
  }

  inline void makeST_hf(const float hfTowers[nHfEta / 2][nHfPhi], GCTsupertower_t supertower_return[nSTEta][nSTPhi]) {
    float et_sumEta[nSTEta][nSTPhi][3];
    float stripEta[nSTEta][nSTPhi][3];
    float stripPhi[nSTEta][nSTPhi][3];

    int index_i = 0;  // 5th and 6th ST to be set 0
    int index_j = 0;
    for (int i = 0; i < nHfEta / 2; i += 3) {  // 12 eta to 4 super towers
      index_j = 0;
      for (int j = 0; j < nHfPhi; j += 3) {  // 72 phi to 24 super towers
        stripEta[index_i][index_j][0] = hfTowers[i][j] + hfTowers[i][j + 1] + hfTowers[i][j + 2];
        stripEta[index_i][index_j][1] = hfTowers[i + 1][j] + hfTowers[i + 1][j + 1] + hfTowers[i + 1][j + 2];
        stripEta[index_i][index_j][2] = hfTowers[i + 2][j] + hfTowers[i + 2][j + 1] + hfTowers[i + 2][j + 2];
        stripPhi[index_i][index_j][0] = hfTowers[i][j] + hfTowers[i + 1][j] + hfTowers[i + 2][j];
        stripPhi[index_i][index_j][1] = hfTowers[i][j + 1] + hfTowers[i + 1][j + 1] + hfTowers[i + 2][j + 1];
        stripPhi[index_i][index_j][2] = hfTowers[i][j + 2] + hfTowers[i + 1][j + 2] + hfTowers[i + 2][j + 2];
        for (int k = 0; k < 3; k++) {
          et_sumEta[index_i][index_j][k] = hfTowers[i + k][j] + hfTowers[i + k][j + 1] + hfTowers[i + k][j + 2];
        }
        index_j++;
      }
      index_i++;
    }

    for (int i = 0; i < nSTEta; i++) {
      for (int j = 0; j < nSTPhi; j++) {
        GCTsupertower_t temp;
        temp.et = 0;
        temp.eta = 0;
        temp.phi = 0;
        temp.towerEta = 0;
        temp.towerPhi = 0;
        temp.towerEt = 0;
        temp.centerEta = 0;
        temp.centerPhi = 0;
        if (i < 4) {
          float supertowerEt = et_sumEta[i][j][0] + et_sumEta[i][j][1] + et_sumEta[i][j][2];
          temp.et = supertowerEt;
          temp.eta = i;
          temp.phi = j;
          int peakEta = getPeakBinOf3(stripEta[i][j][0], stripEta[i][j][1], stripEta[i][j][2]);
          temp.towerEta = peakEta;
          int peakPhi = getPeakBinOf3(stripPhi[i][j][0], stripPhi[i][j][1], stripPhi[i][j][2]);
          temp.towerPhi = peakPhi;
          float peakEt = hfTowers[i * 3 + peakEta][j * 3 + peakPhi];
          temp.towerEt = peakEt;
          int cEta = getEtCenterOf3(stripEta[i][j][0], stripEta[i][j][1], stripEta[i][j][2]);
          temp.centerEta = cEta;
          int cPhi = getEtCenterOf3(stripPhi[i][j][0], stripPhi[i][j][1], stripPhi[i][j][2]);
          temp.centerPhi = cPhi;
        }
        supertower_return[i][j] = temp;
      }
    }
  }

  inline GCTsupertower_t bestOf2(const GCTsupertower_t& calotp0, const GCTsupertower_t& calotp1) {
    GCTsupertower_t x;
    x = (calotp0.et > calotp1.et) ? calotp0 : calotp1;
    return x;
  }

  inline GCTsupertower_t getPeakBin24N(const etaStrip_t& etaStrip) {
    GCTsupertower_t best01 = bestOf2(etaStrip.cr[0], etaStrip.cr[1]);
    GCTsupertower_t best23 = bestOf2(etaStrip.cr[2], etaStrip.cr[3]);
    GCTsupertower_t best45 = bestOf2(etaStrip.cr[4], etaStrip.cr[5]);
    GCTsupertower_t best67 = bestOf2(etaStrip.cr[6], etaStrip.cr[7]);
    GCTsupertower_t best89 = bestOf2(etaStrip.cr[8], etaStrip.cr[9]);
    GCTsupertower_t best1011 = bestOf2(etaStrip.cr[10], etaStrip.cr[11]);
    GCTsupertower_t best1213 = bestOf2(etaStrip.cr[12], etaStrip.cr[13]);
    GCTsupertower_t best1415 = bestOf2(etaStrip.cr[14], etaStrip.cr[15]);
    GCTsupertower_t best1617 = bestOf2(etaStrip.cr[16], etaStrip.cr[17]);
    GCTsupertower_t best1819 = bestOf2(etaStrip.cr[18], etaStrip.cr[19]);
    GCTsupertower_t best2021 = bestOf2(etaStrip.cr[20], etaStrip.cr[21]);
    GCTsupertower_t best2223 = bestOf2(etaStrip.cr[22], etaStrip.cr[23]);

    GCTsupertower_t best0123 = bestOf2(best01, best23);
    GCTsupertower_t best4567 = bestOf2(best45, best67);
    GCTsupertower_t best891011 = bestOf2(best89, best1011);
    GCTsupertower_t best12131415 = bestOf2(best1213, best1415);
    GCTsupertower_t best16171819 = bestOf2(best1617, best1819);
    GCTsupertower_t best20212223 = bestOf2(best2021, best2223);

    GCTsupertower_t best01234567 = bestOf2(best0123, best4567);
    GCTsupertower_t best89101112131415 = bestOf2(best891011, best12131415);
    GCTsupertower_t best16to23 = bestOf2(best16171819, best20212223);

    GCTsupertower_t best0to15 = bestOf2(best01234567, best89101112131415);
    GCTsupertower_t bestOf24 = bestOf2(best0to15, best16to23);

    return bestOf24;
  }

  inline towerMax getPeakBin6N(const etaStripPeak_t& etaStrip) {
    towerMax x;

    GCTsupertower_t best01 = bestOf2(etaStrip.pk[0], etaStrip.pk[1]);
    GCTsupertower_t best23 = bestOf2(etaStrip.pk[2], etaStrip.pk[3]);
    GCTsupertower_t best45 = bestOf2(etaStrip.pk[4], etaStrip.pk[5]);

    GCTsupertower_t best0123 = bestOf2(best01, best23);

    GCTsupertower_t bestOf6 = bestOf2(best0123, best45);

    x.energy = bestOf6.et;
    x.phi = bestOf6.phi;
    x.eta = bestOf6.eta;
    x.energyMax = bestOf6.towerEt;
    x.etaMax = bestOf6.towerEta;
    x.phiMax = bestOf6.towerPhi;
    x.etaCenter = bestOf6.centerEta;
    x.phiCenter = bestOf6.centerPhi;
    return x;
  }

  inline jetInfo getJetPosition(GCTsupertower_t temp[nSTEta][nSTPhi]) {
    etaStripPeak_t etaStripPeak;
    jetInfo jet;

    for (int i = 0; i < nSTEta; i++) {
      etaStrip_t test;
      for (int j = 0; j < nSTPhi; j++) {
        test.cr[j] = temp[i][j];
      }
      etaStripPeak.pk[i] = getPeakBin24N(test);
    }

    towerMax peakIn6;
    peakIn6 = getPeakBin6N(etaStripPeak);

    jet.seedEnergy = peakIn6.energy;
    jet.energy = 0;
    jet.tauEt = 0;
    jet.eta = peakIn6.eta;
    jet.phi = peakIn6.phi;
    jet.energyMax = peakIn6.energyMax;
    jet.etaMax = peakIn6.etaMax;        // overwritten in getJetValues
    jet.phiMax = peakIn6.phiMax;        // overwritten in getJetValues
    jet.etaCenter = peakIn6.etaCenter;  // overwritten in getJetValues
    jet.phiCenter = peakIn6.phiCenter;  // overwritten in getJetValues

    return jet;
  }

  inline jetInfo getJetValues(GCTsupertower_t tempX[nSTEta][nSTPhi], int seed_eta, int seed_phi) {
    float temp[nSTEta + 2][nSTPhi + 2];
    float eta_slice[3];
    jetInfo jet_tmp;

    for (int i = 0; i < nSTEta + 2; i++) {
      for (int k = 0; k < nSTPhi + 2; k++) {
        temp[i][k] = 0;
      }
    }

    for (int i = 0; i < nSTEta; i++) {
      for (int k = 0; k < nSTPhi; k++) {
        temp[i + 1][k + 1] = tempX[i][k].et;
      }
    }

    int seed_eta1, seed_phi1;

    seed_eta1 = seed_eta;  //to start from corner
    seed_phi1 = seed_phi;  //to start from corner
    float tmp1, tmp2, tmp3;

    for (int j = 0; j < nSTEta; j++) {
      for (int k = 0; k < nSTPhi; k++) {
        if (j == seed_eta1 && k == seed_phi1) {
          for (int m = 0; m < 3; m++) {
            tmp1 = temp[j + m][k];
            tmp2 = temp[j + m][k + 1];
            tmp3 = temp[j + m][k + 2];
            eta_slice[m] = tmp1 + tmp2 + tmp3;
          }
        }
      }
    }

    jet_tmp.energy = eta_slice[0] + eta_slice[1] + eta_slice[2];
    jet_tmp.tauEt = eta_slice[1];  //set tau Pt to be sum of ST energies in center eta slice */
    // To find the jet centre: note that seed supertower is always (1, 1)
    jet_tmp.etaCenter =
        3 * seed_eta + tempX[seed_eta][seed_phi].centerEta;  //this is the ET weighted eta centre of the ST
    jet_tmp.phiCenter =
        3 * seed_phi + tempX[seed_eta][seed_phi].centerPhi;  //this is the ET weighted phi centre of the ST
    jet_tmp.etaMax = 3 * seed_eta + tempX[seed_eta][seed_phi].towerEta;
    jet_tmp.phiMax = 3 * seed_phi + tempX[seed_eta][seed_phi].towerPhi;

    // set the used supertower ets to 0
    for (int i = 0; i < nSTEta; i++) {
      if (i + 1 >= seed_eta && i <= seed_eta + 1) {
        for (int k = 0; k < nSTPhi; k++) {
          if (k + 1 >= seed_phi && k <= seed_phi + 1)
            tempX[i][k].et = 0;
        }
      }
    }

    return jet_tmp;
  }

  inline jetInfo getRegion(GCTsupertower_t temp[nSTEta][nSTPhi]) {
    jetInfo jet_tmp, jet;
    jet_tmp = getJetPosition(temp);
    int seed_phi = jet_tmp.phi;
    int seed_eta = jet_tmp.eta;
    float seed_energy = jet_tmp.seedEnergy;
    jet = getJetValues(temp, seed_eta, seed_phi);
    if (seed_energy > 10.) {  // suppress <= 10 GeV ST as seed
      jet_tmp.energy = jet.energy;
      jet_tmp.tauEt = jet.tauEt;
    } else {
      jet_tmp.energy = 0.;
      jet_tmp.tauEt = 0.;
    }
    jet_tmp.etaCenter = jet.etaCenter;  // this is the ET weighted eta centre of the ST
    jet_tmp.phiCenter = jet.phiCenter;  // this is the ET weighted eta centre of the ST
    jet_tmp.etaMax = jet.etaMax;        // this is the leading tower eta in the ST
    jet_tmp.phiMax = jet.phiMax;        // this is the leading tower phi in the ST
    return jet_tmp;
  }

  inline bool compareByEt(l1tp2::Phase2L1CaloJet i, l1tp2::Phase2L1CaloJet j) { return (i.jetEt() > j.jetEt()); };

}  // namespace gctobj

#endif
