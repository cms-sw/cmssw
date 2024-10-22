#ifndef Geometry_HcalTowerAlgo_HcalDDDSimConstants_h
#define Geometry_HcalTowerAlgo_HcalDDDSimConstants_h

/** \class HcalDDDSimConstants
 *
 * this class reads the constant section of
 * the hcal-sim-numbering xml-file
 *  
 * \author Sunanda Banerjee, SINP <sunanda.banerjee@cern.ch>
 *
 */

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "Geometry/HcalCommonData/interface/HcalCellType.h"
#include "Geometry/HcalCommonData/interface/HcalLayerDepthMap.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

class HcalDDDSimConstants {
public:
  HcalDDDSimConstants(const HcalParameters* hp);
  ~HcalDDDSimConstants();

  HcalCellType::HcalCell cell(
      const int& det, const int& zside, const int& depth, const int& etaR, const int& iphi) const;
  int findDepth(const int& det, const int& eta, const int& phi, const int& zside, const int& lay) const;
  unsigned int findLayer(const int& layer, const std::vector<HcalParameters::LayerItem>& layerGroup) const;
  std::vector<std::pair<double, double> > getConstHBHE(const int& type) const;
  int getDepthEta16(const int& det, const int& phi, const int& zside) const;
  int getDepthEta16M(const int& det) const;
  int getDepthEta29(const int& phi, const int& zside, const int& i) const;
  int getDepthEta29M(const int& i, const bool& planOne) const;
  std::pair<int, double> getDetEta(const double& eta, const int& depth) const;
  int getEta(const int& det, const int& lay, const double& hetaR) const;
  std::pair<int, int> getEtaDepth(
      const int& det, int etaR, const int& phi, const int& zside, int depth, const int& lay) const;
  double getEtaHO(const double& etaR, const double& x, const double& y, const double& z) const;
  std::pair<int, int> getiEtaRange(const int& i) const { return std::pair<int, int>(hpar->etaMin[i], hpar->etaMax[i]); }
  const std::vector<double>& getEtaTableHF() const { return hpar->etaTableHF; }
  const std::vector<double>& getGparHF() const { return hpar->gparHF; }
  const std::vector<HcalDetId>& getIdHF2QIE() const { return idHF2QIE; }
  double getLayer0Wt(const int& det, const int& phi, const int& zside) const;
  int getFrontLayer(const int& det, const int& eta) const;
  int getLastLayer(const int& det, const int& eta) const;
  int getLayerFront(const int& det, const int& eta, const int& phi, const int& zside, const int& depth) const;
  int getLayerBack(const int& det, const int& eta, const int& phi, const int& zside, const int& depth) const;
  int getLayerMax(const int& eta, const int& depth) const;
  int getMaxDepth(const int& type) const { return maxDepth[type]; }
  int getMaxDepth(const int& det, const int& eta, const int& phi, const int& zside, const bool& partialOnly) const;
  std::pair<int, int> getMaxDepthDet(const int& i) const { return ((i == 1) ? depthMaxDf_ : depthMaxSp_); }
  int getMinDepth(const int& det, const int& eta, const int& phi, const int& zside, const bool& partialOnly) const;
  std::pair<int, int> getModHalfHBHE(const int& type) const;
  std::pair<double, double> getPhiCons(const int& det, const int& ieta) const;
  std::vector<std::pair<int, double> > getPhis(const int& subdet, const int& ieta) const;
  const std::vector<double>& getPhiTableHF() const { return hpar->phitable; }
  const std::vector<double>& getRTableHF() const { return hpar->rTable; }
  std::vector<HcalCellType> HcalCellTypes() const;
  std::vector<HcalCellType> HcalCellTypes(const HcalSubdetector&, int ieta = -1, int depth = -1) const;
  bool isBH() const { return isBH_; }
  bool isHE() const { return (hpar->etaMax[1] > hpar->etaMin[1]); }
  const HcalLayerDepthMap* ldMap() const { return &ldmap_; }
  int maxHFDepth(const int& ieta, const int& iphi) const;
  unsigned int numberOfCells(const HcalSubdetector&) const;
  const HcalParameters* parameter() const { return hpar; }
  int phiNumber(const int& phi, const int& unit) const;
  void printTiles() const;
  int unitPhi(const int& det, const int& etaR) const;
  int unitPhi(const double& dphi) const;

private:
  static const int nDepthMax = 9;
  static const int maxLayer_ = 18;
  static const int maxLayerHB_ = 16;

  void initialize();
  double deltaEta(const int& det, const int& eta, const int& depth) const;
  double getEta(const int& det, const int& etaR, const int& zside, int depth = 1) const;
  double getEta(const double& r, const double& z) const;
  int getShift(const HcalSubdetector& subdet, const int& depth) const;
  double getGain(const HcalSubdetector& subdet, const int& depth) const;
  void printTileHB(const int& eta, const int& phi, const int& zside, const int& depth) const;
  void printTileHE(const int& eta, const int& phi, const int& zside, const int& depth) const;
  unsigned int layerGroupSize(int eta) const;
  unsigned int layerGroup(int eta, int i) const;
  unsigned int layerGroup(int det, int eta, int phi, int zside, int i) const;

  const HcalParameters* hpar;
  HcalLayerDepthMap ldmap_;

  std::vector<int> maxDepth;                     // Maximum depths in HB/HE/HF/HO
  int nEta;                                      // Number of bins in eta for HB and HE
  int nR;                                        // Number of bins in r
  int nPhiF;                                     // Number of bins in phitable
  std::vector<int> depths[nDepthMax];            // Maximum layer number for a depth
  int nDepth;                                    // Number of bins in depth0
  int nzHB, nmodHB;                              // Number of halves and modules in HB
  int nzHE, nmodHE;                              // Number of halves and modules in HE
  double etaHO[4], rminHO;                       // eta in HO ring boundaries
  double zVcal;                                  // Z-position  of the front of HF
  double dzVcal;                                 // Half length of the HF
  double dlShort;                                // Diference of length between long and short
  int depthEta16[2];                             // depth index of ieta=16 for HBmax, HEMin
  int depthEta29[2];                             // maximum depth index for ieta=29
  int layFHB[2];                                 // first layers in HB (normal, special 16)
  int layFHE[3];                                 // first layers in HE (normal, special 16, 18)
  int layBHB[3];                                 // last layers in HB (normal, special 15, 16)
  int layBHE[4];                                 // last layers in HE (normal, special 16, 17, 18)
  bool isBH_;                                    // if HE is BH
  std::vector<HcalDetId> idHF2QIE;               // DetId's of HF modules with 2 QIE's
  std::pair<int, int> depthMaxDf_, depthMaxSp_;  // (subdet,maximum depth) default,special
};

#endif
