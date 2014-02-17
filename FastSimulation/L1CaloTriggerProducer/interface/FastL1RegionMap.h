#ifndef RecoTauTag_FastL1RegionMap_h
#define RecoTauTag_FastL1RegionMap_h
// -*- C++ -*-
//
// Package:    L1CaloSim
// Class:      FastL1RegionMap
// 
/**\class FastL1RegionMap

 Description: Mapping between DetIds, CaloTower IDs and Region IDs.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chi Nhan Nguyen
//         Created:  Mon Feb 19 13:25:24 CST 2007
// $Id: FastL1RegionMap.h,v 1.3 2009/03/23 11:41:27 chinhan Exp $
//

#include <iostream>
#include <string>
#include <list> 

#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"


//
// This Class is used for the mapping Detector IDs to the region IDs
//
// Region (HF not yet considered!!!)
// 22x18 (ieta x iphi) map:
//
// ieta: 0  1  2  3 ... 21
//       ...
//       ...
//
// iphi: 17 17 17 17 ... 17
//       .  .  .  .  ... .
//       .  .  .  .  ... .
//       2  .  .  .  ... 2
//       1  .  .  .  ... 1
//       0  .  .  .  ... 0
//
// barrel ieta: 7-14
// endcap ieta: 4-6, 15-17
// HF ieta: 0-3, 18-21 // HF not considered yet!
// 
class FastL1RegionMap {

 public:
  FastL1RegionMap();
  ~FastL1RegionMap();

  static FastL1RegionMap* getFastL1RegionMap();

  std::pair<int, int> getRegionEtaPhiIndex(std::pair<int, int> iEtaPhi);
  std::pair<int, int> getRegionEtaPhiIndex(CaloTowerDetId towerId);
  std::pair<int, int> getRegionEtaPhiIndex(int regionId);
  int getRegionIndex(int ieta, int iphi);
  int getRegionIndex(CaloTowerDetId tower);
  int getRegionTowerIndex(std::pair<int, int> iEtaPhi);
  int getRegionTowerIndex(int ieta, int iphi);
  int getRegionTowerIndex(CaloTowerDetId towerId);

  std::pair<double, double> getRegionCenterEtaPhi(int iRgn);

  int getNTower() { return nTower; };
  int getNRegion() { return nRegion; };

  int convertFromECal_to_HCal_iphi(int iphi_ecal);
  int convertFromHCal_to_ECal_iphi(int iphi_hcal);

  void display();

  //CaloTowerDetId getTowerId();

  std::pair<int, int> GetTowerNorthEtaPhi(int ieta, int iphi); 
  std::pair<int, int> GetTowerSouthEtaPhi(int ieta, int iphi); 
  std::pair<int, int> GetTowerWestEtaPhi(int ieta, int iphi); 
  std::pair<int, int> GetTowerEastEtaPhi(int ieta, int iphi); 
  std::pair<int, int> GetTowerNWEtaPhi(int ieta, int iphi); 
  std::pair<int, int> GetTowerNEEtaPhi(int ieta, int iphi); 
  std::pair<int, int> GetTowerSWEtaPhi(int ieta, int iphi); 
  std::pair<int, int> GetTowerSEEtaPhi(int ieta, int iphi); 

 private:
  static FastL1RegionMap* theInstance;

  int nTower;
  int nRegion;


};

#endif
