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
    int  eta, phi, depth;
    HcalID(int et=0, int fi=0, int d=0) : eta(et), phi(fi), depth(d) {}
  };
  HcalID                    getHCID(int subdet, int ieta, int iphi, int lay,
				    int idepth) const;
  std::vector<int>          getDepth(const int i) const {return layerGroup[i];}
  std::vector<double>       getEtaTable() const {return etaTable;}
  std::pair<double,double>  getEtaLimit(const int i) const 
    {return std::pair<double,double>(etaTable[i],etaTable[i+1]);}
  int                       getNEta() const {return nEta;}
  double                    getPhiBin(const int i) const {return phibin[i];}
  double                    getPhiOff(const int i) const {return phioff[i];}
       
private:
  HcalDDDRecConstants();
  void loadSpecPars(const DDFilteredView& fv);
  void loadSimConst(const HcalDDDSimConstants& hcons);
  std::vector<double> getDDDArray(const char *, const DDsvalues_type &, int &) const;

  static const int nEtaMax=100;
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
};

#endif
