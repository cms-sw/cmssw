#ifndef Geometry_HcalTowerAlgo_HcalDDDSimConstants_h
#define Geometry_HcalTowerAlgo_HcalDDDSimConstants_h

/** \class HcalDDDSimConstants
 *
 * this class reads the constant section of
 * the hcal-sim-numbering xml-file
 *  
 *  $Date: 2013/12/25 00:06:50 $
 * \author Sunanda Banerjee, SINP <sunanda.banerjee@cern.ch>
 *
 */

#include<string>
#include<vector>
#include<iostream>
#include<iomanip>

#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "Geometry/HcalCommonData/interface/HcalCellType.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

class HcalDDDSimConstants {

public:

  HcalDDDSimConstants(const HcalParameters* hp);
  ~HcalDDDSimConstants();

  HcalCellType::HcalCell    cell(int det, int zside, int depth, int etaR, 
				 int iphi) const;
  std::vector<std::pair<double,double> > getConstHBHE(const int type) const;
  std::pair<int,double>     getDetEta(double eta, int depth);
  int                       getEta(int det,int lay, double hetaR);
  std::pair<int,int>        getEtaDepth(int det, int etaR, int phi, int depth,
					int lay);
  double                    getEtaHO(double& etaR, double& x, double& y, 
				     double& z) const;
  std::pair<int,int>        getiEtaRange(const int i) const {return std::pair<int,int>(etaMin[i],etaMax[i]);}
  std::vector<double>       getEtaTableHF() const;
  std::vector<double>       getGparHF() const {return hpar->gparHF;}
  std::vector<double>       getLayer0Wt() const {return hpar->Layer0Wt;}
  std::pair<int,int>        getModHalfHBHE(const int type) const;
  std::vector<int>          getNOff() const {return hpar->noff;}
  double                    getPhiBin(const int i) const {return hpar->phibin[i];}
  std::pair<double,double>  getPhiCons(int det, int ieta);
  double                    getPhiOff(const int i) const {return hpar->phioff[i];}
  std::vector<double>       getPhiOffs()  const {return hpar->phioff;}
  std::vector<double>       getPhiTable() const {return hpar->phibin;}
  std::vector<double>       getPhiTableHF() const {return hpar->phitable;}
  std::vector<double>       getRTableHF() const {return hpar->rTable;}
  std::vector<HcalCellType> HcalCellTypes() const;
  std::vector<HcalCellType> HcalCellTypes(HcalSubdetector, int ieta=-1,
					  int depth=-1) const;
  int                       getMaxDepth(const int type) const {return maxDepth[type];}
  unsigned int              numberOfCells(HcalSubdetector) const;
  int                       phiNumber(int phi, int unit) const;
  void                      printTiles() const;
  int                       unitPhi(int det, int etaR) const;
  int                       unitPhi(double dphi) const; 
       
private:
  void                      initialize();
  unsigned int              findLayer(int layer, const std::vector<HcalParameters::LayerItem>& layerGroup) const;
  double                    deltaEta(int det, int eta, int depth) const;
  double                    getEta(int det, int etaR, int zside, int depth=1) const;
  double                    getEta(double r, double z) const;
  int                       getShift(HcalSubdetector subdet, int depth) const;
  double                    getGain (HcalSubdetector subdet, int depth) const;
  void                      printTileHB(int eta, int depth) const;
  void                      printTileHE(int eta, int depth) const;
  
  const HcalParameters* hpar;
  static const int    nEtaMax=100;
  static const int    nDepthMax=9;
  std::vector<int>    maxDepth; // Maximum depths in HB/HE/HF/HO
  int                 nEta;     // Number of bins in eta for HB and HE
  int                 nR;       // Number of bins in r
  int                 nPhiF;    // Number of bins in phitable
  std::vector<int>    etaMin, etaMax;      // Eta ranges in HB/HE/HF/HO
  std::vector<int>    layerGroup[nEtaMax]; // Depth index for a given layer
  std::vector<int>    depths[nDepthMax];   // Maximum layer number for a depth 
  int                 nDepth;   // Number of bins in depth0
  int                 nzHB, nmodHB;     // Number of halves and modules in HB
  int                 nzHE, nmodHE;     // Number of halves and modules in HE
  double              etaHO[4], rminHO; // eta in HO ring boundaries
  double              zVcal;    // Z-position  of the front of HF
  double              dzVcal;   // Half length of the HF
  double              dlShort;  // Diference of length between long and short
};

#endif
