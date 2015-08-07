#ifndef Geometry_HcalTowerAlgo_HcalDDDRecConstants_h
#define Geometry_HcalTowerAlgo_HcalDDDRecConstants_h

/** \class HcalDDDRecConstants
 *
 * this class reads the constant section of
 * the hcal-sim-numbering xml-file
 *  
 * \author Sunanda Banerjee, SINP <sunanda.banerjee@cern.ch>
 *
 */

#include<string>
#include<vector>
#include<iostream>

#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "Geometry/HcalCommonData/interface/HcalCellType.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

class HcalDDDRecConstants {

public:

  HcalDDDRecConstants(const HcalParameters* hp, const HcalDDDSimConstants& hc);
  ~HcalDDDRecConstants();

  struct HcalID {
    int    subdet, eta, phi, depth;
    HcalID(int sub=0, int et=0, int fi=0, int d=0) : subdet(sub), eta(et),
						     phi(fi), depth(d) {}
  };
  struct HcalEtaBin {
    int    ieta, nPhi, depthStart;
    double etaMin, etaMax, phi0, dphi;
    std::vector<std::pair<int, int> > layer;
    HcalEtaBin(int eta=0, double et1=0, double et2=0, int nf=0, double fi0=0,
	       double df=0) : ieta(eta), nPhi(nf),depthStart(0), etaMin(et1), 
			      etaMax(et2), phi0(fi0), dphi(df) {}
  };
  struct HcalActiveLength {
    int    ieta, depth;
    double eta, thick;
    HcalActiveLength(int ie=0, int d=0, double et=0, 
		     double t=0) : ieta(ie), depth(d), eta(et), thick(t) {}
  };

  std::vector<std::pair<double,double> > getConstHBHE(const int type) const {
    if      (type == 0) return gconsHB;
    else if (type == 1) return gconsHE;
    else {std::vector<std::pair<double,double> > gcons; return gcons;}
  }
  const std::vector<int> &  getDepth(const unsigned int i) const;
  std::vector<HcalEtaBin>   getEtaBins(const int itype) const;
  std::pair<double,double>  getEtaPhi(int subdet, int ieta, int iphi) const;
  std::pair<int,int>        getEtaRange(const int i) const
    {return std::pair<int,int>(iEtaMin[i],iEtaMax[i]);}
  const std::vector<double> &      getEtaTable()   const {return etaTable;}
  const std::vector<double> &      getEtaTableHF() const {return hpar->etaTableHF;}
  std::pair<double,double>  getEtaLimit(const int i) const 
    {return std::pair<double,double>(etaTable[i],etaTable[i+1]);}
  HcalID                    getHCID(int subdet, int ieta, int iphi, int lay,
				    int idepth) const;
  int                       getMaxDepth(const int type) const {return maxDepth[type];}
  int                       getNEta() const {return hpar->etagroup.size();}
  double                    getPhiBin(const int i) const {return phibin[i];}
  double                    getPhiOff(const int i) const {return hpar->phioff[i];}
  const std::vector<double> &      getPhiOffs()    const {return hpar->phioff;}
  const std::vector<double> &      getPhiTable()   const {return phibin;}
  const std::vector<double> &      getPhiTableHF() const {return hpar->phitable;}
  double                    getRZ(int subdet, int ieta, int depth) const;
  std::vector<HcalActiveLength>    getThickActive(const int type) const;
  int                       getTopoMode() const {return hpar->topologyMode;}
  std::vector<HcalCellType> HcalCellTypes(HcalSubdetector) const;
  unsigned int              numberOfCells(HcalSubdetector) const;
  unsigned int              nCells(HcalSubdetector) const;
  unsigned int              nCells() const;
       
private:
  void                      initialize(void);
  unsigned int              layerGroupSize( unsigned int eta ) const;
  unsigned int              layerGroup( unsigned int eta, unsigned int i ) const;

  const HcalParameters      *hpar;
  const HcalDDDSimConstants &hcons;
  std::vector<std::pair<int,int> > etaSimValu; // eta ranges at Sim stage
  std::vector<double> etaTable;   // Eta table (HB+HE)
  std::vector<int>    ietaMap;    // Map Sim level ieta to Rec level ieta
  std::vector<int>    iEtaMin, iEtaMax; // Minimum and maximum eta
  std::vector<int>    maxDepth;   // Maximum depth in HB/HE/HF/HO 
  std::vector<double> phibin;     // Phi step for all eta bins (HB, HE, HO)
  std::vector<int>    phiUnitS;   // Phi unit at SIM stage
  std::vector<std::pair<double,double> > gconsHB; // Geometry constatnts HB
  std::vector<std::pair<double,double> > gconsHE; // Geometry constatnts HE
  int                 nModule[2], nHalves[2];     // Modules, Halves for HB/HE
};

#endif
