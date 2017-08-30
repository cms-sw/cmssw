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

#include<map>
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
    int    ieta, zside, depthStart;
    double dphi, etaMin, etaMax;
    std::vector<std::pair<int, int> > layer;
    std::vector<std::pair<int,double> > phis;
    HcalEtaBin(int eta=0, int zs=1, double dfi=0, double et1=0, 
  	       double et2=0) : ieta(eta), zside(zs), depthStart(0), dphi(dfi), 
                               etaMin(et1), etaMax(et2) {}
  };
  struct HcalActiveLength {
    int    ieta, depth, zside, stype;
    double eta, thick;
    std::vector<int> iphis;
    HcalActiveLength(int ie=0, int d=0, int z=0, int s=0, double et=0, 
		     double t=0) : ieta(ie), depth(d), zside(z), stype(s),
                                   eta(et), thick(t) {}
  };
  struct HFCellParameters {
    int    ieta, depth, firstPhi, stepPhi, nPhi;
    double rMin, rMax;
    HFCellParameters(int ie=0, int d=1, int ffi=1, int sfi=2, int nfi=36,
		     double r1=0, double r2=0) : ieta(ie), depth(d), 
                                                 firstPhi(ffi), stepPhi(sfi),
                                                 nPhi(nfi), rMin(r1), rMax(r2) {}
  };

  std::vector<std::pair<double,double> > getConstHBHE(const int& type) const {
    if      (type == 0) return gconsHB;
    else if (type == 1) return gconsHE;
    else {std::vector<std::pair<double,double> > gcons; return gcons;}
  }
  std::vector<int>          getDepth(const int& det, const int& phi, 
				     const int& zside, const unsigned int& eta) const;
  std::vector<int>          getDepth(const unsigned int& eta, const bool& extra) const;
  int                       getDepthEta16(const int& det, const int& iphi, 
					  const int& zside) const {return hcons.getDepthEta16(det,iphi,zside);}
  std::vector<HcalEtaBin>   getEtaBins(const int& itype) const;
  std::pair<double,double>  getEtaPhi(const int& subdet, const int& ieta, const int& iphi) const;
  std::pair<int,int>        getEtaRange(const int& i) const
    {return std::pair<int,int>(iEtaMin[i],iEtaMax[i]);}
  const std::vector<double> &      getEtaTable()   const {return etaTable;}
  const std::vector<double> &      getEtaTableHF() const {return hpar->etaTableHF;}
  std::pair<double,double>  getEtaLimit(const int& i) const 
    {return std::pair<double,double>(etaTable[i],etaTable[i+1]);}
  HcalID                    getHCID(int subdet, int ieta, int iphi, int lay, int idepth) const;
  std::vector<HFCellParameters>    getHFCellParameters() const;
  void                      getLayerDepth(const int& ieta, std::map<int,int>& layers) const;
  int                       getLayerFront(const int& det, const int& eta, const int& phi,
					  const int& depth) const;
  double                    getLayer0Wt(const int& det, const int& phi,
					const int& zside) const {return hcons.getLayer0Wt(det,phi,zside);}
  int                       getMaxDepth(const int& type) const {return maxDepth[type];}
  int                       getMaxDepth(const int& itype, const int& ieta,
					const int& iphi,  const int& zside) const;
  int                       getMinDepth(const int& itype, const int& ieta,
					const int& iphi,  const int& zside) const;
  int                       getNEta() const {return hpar->etagroup.size();}
  int                       getNoff(const int& i) const {return hpar->noff[i];}
  int                       getNPhi(const int& type) const {return nPhiBins[type];}
  double                    getPhiBin(const int& i) const {return phibin[i];}
  double                    getPhiOff(const int& i) const {return hpar->phioff[i];}
  const std::vector<double> &      getPhiOffs()    const {return hpar->phioff;}
  std::vector<std::pair<int,double> > getPhis(const int& subdet, const int& ieta) const;
  const std::vector<double> &      getPhiTable()   const {return phibin;}
  const std::vector<double> &      getPhiTableHF() const {return hpar->phitable;}
  int                       getPhiZOne(std::vector<std::pair<int,int> >& phiz) const;
  double                    getRZ(const int& subdet, const int& ieta, const int& depth) const;
  double                    getRZ(const int& subdet, const int& ieta, const int& iphi,
				  const int& depth) const;
  double                    getRZ(const int& subdet, const int& layer) const;
  std::vector<HcalActiveLength>    getThickActive(const int& type) const;
  int                       getTopoMode() const {return ((hpar->topologyMode)&0xFF);}
  int                       getTriggerMode() const {return (((hpar->topologyMode)>>8)&0xFF);}
  std::vector<HcalCellType> HcalCellTypes(HcalSubdetector) const;
  bool                      isBH() const {return hcons.isBH();}
  bool                      isPlan1(const HcalDetId& id) const { return detIdSp_.find(id) != detIdSp_.end(); };
  int                       maxHFDepth(int ieta, int iphi) const {return hcons.maxHFDepth(ieta,iphi);}
  unsigned int              numberOfCells(HcalSubdetector) const;
  unsigned int              nCells(HcalSubdetector) const;
  unsigned int              nCells() const;
  HcalDetId                 mergedDepthDetId(const HcalDetId& id) const;
  HcalDetId                 idFront(const HcalDetId& id) const;
  HcalDetId                 idBack (const HcalDetId& id) const;
  void                      unmergeDepthDetId(const HcalDetId& id,
					      std::vector<HcalDetId>& ids) const;
  void                      specialRBXHBHE(const std::vector<HcalDetId>&,
					   std::vector<HcalDetId> &) const;
  bool                      specialRBXHBHE(bool flag,
					   std::vector<HcalDetId> &) const;
  bool                      withSpecialRBXHBHE() const {return (hcons.ldMap()->getSubdet() != 0);}
  bool                      isPlan1ToBeMergedId(const HcalDetId& id) const { return detIdSp_.find(id) != detIdSp_.end(); };
  bool                      isPlan1MergedId(const HcalDetId& id) const { return detIdSpR_.find(id) != detIdSpR_.end(); };
  const HcalDDDSimConstants* dddConstants() const {return &hcons;}
       
private:

  void                      getOneEtaBin(HcalSubdetector subdet, int ieta, int zside,
					 std::vector<std::pair<int,double>>& phis,
					 std::map<int,int>& layers, bool planOne,
					 std::vector<HcalDDDRecConstants::HcalEtaBin>& bins) const;
  void                      initialize(void);
  unsigned int              layerGroupSize(int eta) const;
  unsigned int              layerGroup(int eta, int i) const;

  static const int           maxLayer_=18;
  static const int           maxLayerHB_=16;
  const HcalParameters      *hpar;
  const HcalDDDSimConstants &hcons;
  std::vector<std::pair<int,int> > etaSimValu; // eta ranges at Sim stage
  std::vector<double> etaTable;   // Eta table (HB+HE)
  std::vector<int>    ietaMap;    // Map Sim level ieta to Rec level ieta
  std::vector<int>    iEtaMin, iEtaMax; // Minimum and maximum eta
  std::vector<int>    maxDepth;   // Maximum depth in HB/HE/HF/HO 
  std::vector<int>    nPhiBins;   // Number of phi bis for HB/HE/HF/HO
  std::vector<double> phibin;     // Phi step for all eta bins (HB, HE, HO)
  std::vector<int>    phiUnitS;   // Phi unit at SIM stage
  std::vector<std::pair<double,double> > gconsHB; // Geometry constatnts HB
  std::vector<std::pair<double,double> > gconsHE; // Geometry constatnts HE
  int                 nModule[2], nHalves[2];     // Modules, Halves for HB/HE
  std::pair<int,int>  depthMaxDf_, depthMaxSp_; // (subdet,maximum depth) default,special
  std::map<HcalDetId,HcalDetId> detIdSp_;       // Map of Id's for special RBX
  std::map<HcalDetId,std::vector<HcalDetId>> detIdSpR_; // Reverse map for RBX
};

#endif
