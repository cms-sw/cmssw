#ifndef HcalCommonData_HcalDDDRecConstants_h
#define HcalCommonData_HcalDDDRecConstants_h

/** \class HcalDDDRecConstants
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

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "Geometry/HcalCommonData/interface/HcalCellType.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

class DDCompactView;    
class DDFilteredView;

class HcalDDDRecConstants {

public:

  HcalDDDRecConstants(const DDCompactView& cpv, 
		      const HcalDDDSimConstants& hcons);
  ~HcalDDDRecConstants();

  struct HcalID {
    int    eta, phi, depth;
    HcalID(int et=0, int fi=0, int d=0) : eta(et), phi(fi), depth(d) {}
  };
  struct HcalEtaBin {
    int    nPhi, depthStart;
    double etaMin, etaMax, phi0, dphi;
    std::vector<std::pair<int, int> > layer;
    HcalEtaBin(double et1=0, double et2=0, int nf=0, double fi0=0,
	       double df=0) : nPhi(nf),depthStart(0), etaMin(et1), etaMax(et2),
			      phi0(fi0), dphi(df) {}
  };

  std::vector<int>          getDepth(const int i) const {return layerGroup[i];}
  std::vector<HcalEtaBin>   getEtaBins(const int itype) const;
  std::vector<double>       getEtaTable() const {return etaTable;}
  std::pair<double,double>  getEtaLimit(const int i) const 
    {return std::pair<double,double>(etaTable[i],etaTable[i+1]);}
  HcalID                    getHCID(int subdet, int ieta, int iphi, int lay,
				    int idepth) const;
  int                       getMaxDepth(const int type) const {return maxDepth[type];}
  int                       getNEta() const {return nEta;}
  double                    getPhiBin(const int i) const {return phibin[i];}
  double                    getPhiOff(const int i) const {return phioff[i];}
  std::string               getTopoMode() const {return modeTopo;}
       
private:
  HcalDDDRecConstants();
  void loadSpecPars(const DDFilteredView& fv);
  void loadSimConst();
  std::vector<double> getDDDArray(const char *, const DDsvalues_type &, int &) const;
  std::string getDDDString(const std::string &, const DDsvalues_type &) const;

  static const int nEtaMax=100;
  const HcalDDDSimConstants *hcons;
  std::string         modeTopo;   // Mode for topology
  std::vector<double> phioff;     // Phi offset for barrel, endcap, forward
  std::vector<int>    etaGroup;   // Eta Grouping
  std::vector<double> etaTable;   // Eta table 
  std::vector<int>    ietaMap;    // Map Sim level ieta to Rec level ieta
  int                 iEtaMin[4], iEtaMax[4]; // Minimum and maximum eta
  int                 maxDepth[4];// Maximum depth in HB/HE/HF/HO           
  int                 nEta;       // Number of bins in eta for HB and HE
  std::vector<int>    phiGroup;   // Eta Grouping
  std::vector<double> phibin;     // Phi step for all eta bins
  std::vector<int>    layerGroup[nEtaMax];
  std::vector<int>    nOff;     // Speical eta bin #'s in barrel and endcap
  std::vector<std::pair<double,double> > gconsHB; // Geometry constatnts HB
  std::vector<std::pair<double,double> > gconsHE; // Geometry constatnts HE
  int                 nModule[2], nHalves[2];     // Modules, Halves for HB/HE
};

#endif
