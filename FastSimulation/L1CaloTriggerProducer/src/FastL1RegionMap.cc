// -*- C++ -*-
//
// Package:    FastL1CaloSim
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
// $Id: FastL1RegionMap.cc,v 1.12 2009/05/05 09:08:39 elmer Exp $
//


#include "FastSimulation/L1CaloTriggerProducer/interface/FastL1RegionMap.h"
#include <cstdlib>

FastL1RegionMap::FastL1RegionMap()
{
  nTower = 4608;
  nRegion = 396;

}

FastL1RegionMap* FastL1RegionMap::theInstance = 0;

FastL1RegionMap*
FastL1RegionMap::getFastL1RegionMap()
{
  if(theInstance == 0)
    {
      theInstance = new FastL1RegionMap();
    }
  return theInstance;
}

// Region ID from DetId
std::pair<int, int> 
FastL1RegionMap::getRegionEtaPhiIndex(CaloTowerDetId tower)
{

  return FastL1RegionMap::getRegionEtaPhiIndex(std::pair<int, int>(tower.ieta(), tower.iphi()));
}

// Region-Tower ID from DetId
int 
FastL1RegionMap::getRegionTowerIndex(CaloTowerDetId tower)
{
  return FastL1RegionMap::getRegionTowerIndex(std::pair<int, int>(tower.ieta(), tower.iphi()));
}


// Mapping of calotowers and regions
// input:  calotower ieta,iphi
// output: region ieta,iphi
std::pair<int, int> 
FastL1RegionMap::getRegionEtaPhiIndex(std::pair<int, int> EtaPhi)
{
  //// *** TDR ***
  // barrel: 2x(17x72) [eta,phi] -> 2x1224 (etaid: +/- 1-17  phiid: 1-72)
  // endcap: 2x(11x72) [eta,phi] -> 2x792  (etaid: +/- 18-28 phiid: 1-72)
  // forward: 2x(4x18) [eta,phi] -> 2x72   (etaid: +/- 29-32 phiid: 1-18)
  //// *** Option1 ***
  // barrel:  2x(20x72) [eta,phi] -> 2x1440 (etaid: +/- 1-20  phiid: 1-72)
  // endcap:  2x(18x36) [eta,phi] -> 2x648  (etaid: +/- 21-38 phiid: 1-36)
  // forward: 2x(3x18)  [eta,phi] -> 2x54   (etaid: +/- 40-41 phiid: 1-18)
  //// *** Option2 ***
  // barrel: 2x(20x72) [eta,phi] -> 2x1440 (etaid: +/- 1-20  phiid: 1-72 interv: 1)
  // endcap: 2x(19x36) [eta,phi] -> 2x684  (etaid: +/- 21-39 phiid: 1-72 interv: 2)
  // forward: 2x(2x18) [eta,phi] -> 2x36   (etaid: +/- 40-41 phiid: 1-72 interv: 4)
  //// *** Option3 ***
  // barrel: 2x(20x72) [eta,phi] -> 2x1440 (etaid: +/- 1-20  phiid: 1-72 interv: 1)
  // endcap: 2x(18x36) [eta,phi] -> 2x684  (etaid: +/- 21-28 phiid: 1-72 interv: 2)
  // forward: 2x(2x18) [eta,phi] -> 2x36   (etaid: +/- 29-32 phiid: 1-72 interv: 4)

  int iTwrEta = EtaPhi.first;
  int iTwrPhi = EtaPhi.second;

  //iTwrPhi = convertFromECal_to_HCal_iphi(iTwrPhi);

  int iphi=999; // 0-17 
  int ieta=999; // 0-21 (barrel: 6-15, endcap: 4,5,16,17, HF: 0-3,18-21??)

  // Right now: only barrel/endcap parts work!!!
  if (abs(iTwrEta)<=28) {

    //int isub; // 0-15 4x4 region matrix 
    
    iphi = ((iTwrPhi + 1) / 4) % 18;
    
    if (iTwrEta > 0) {
      ieta = (iTwrEta - 1) / 4  + 11;
    } else {
      ieta = (iTwrEta + 1) / 4  + 10;
    }
  }

  // Test HF!!!
  if (abs(iTwrEta)>=29 && abs(iTwrEta)<=32) {
    iphi = ((iTwrPhi + 1) / 4) % 18;

    if (iTwrEta == 29) {
      ieta = 18;
    } 

    if (iTwrEta == 30) {
      ieta = 19;
    } 

    if (iTwrEta == 31) {
      ieta = 20;
    } 

    if (iTwrEta == 32) {
      ieta = 21;
    } 

    if (iTwrEta == -29) {
      ieta = 3;
    } 

    if (iTwrEta == -30) {
      ieta = 2;
    } 

    if (iTwrEta == -31) {
      ieta = 1;
    } 

    if (iTwrEta == -32) {
      ieta = 0;
    } 

    /*
    if (iTwrEta >= 29 && iTwrEta <= 32) {
      ieta = 18;
    } 
    if (iTwrEta >= 33 && iTwrEta <= 35) {
      ieta = 19;
    } 
    if (iTwrEta >= 36 && iTwrEta <= 38) {
      ieta = 20;
    } 
    if (iTwrEta >= 39 && iTwrEta <= 41) {
      ieta = 21;
    } 

    if (iTwrEta <= -29 && iTwrEta >= -32) {
      ieta = 3;
    } 
    if (iTwrEta <= -33 && iTwrEta >= -35) {
      ieta = 2;
    } 
    if (iTwrEta <= -36 && iTwrEta >= -38) {
      ieta = 1;
    } 
    if (iTwrEta <= -39 && iTwrEta >= -41) {
      ieta = 0;
    } 
    */
  }

  return std::pair<int, int>(ieta, iphi);
}

// Mapping of calotowers and regions
// input:  calotower ieta,iphi
// output: region isub 0-15 of 4x4 matrix
int 
FastL1RegionMap::getRegionTowerIndex(std::pair<int, int> EtaPhi)
{
  int iTwrEta = EtaPhi.first;
  int iTwrPhi = EtaPhi.second;

  //iTwrPhi = convertFromECal_to_HCal_iphi(iTwrPhi);

  // Right now: only barrel/encap part!!!
  int isub = 999; // 0-15 4x4 region matrix 

  if (abs(iTwrEta)<=41) {
    //if (abs(iTwrEta)<=28) {
    if (iTwrEta > 0) {
      isub = 4*(3 - (iTwrPhi + 1) %4) + ((iTwrEta  - 1)  % 4) ;
    } else {
      isub = 4*(3 - (iTwrPhi + 1) %4) + (3 + (iTwrEta+1)%4) ;
    }
  }

  return isub;
}

int 
FastL1RegionMap::getRegionTowerIndex(int iEta, int iPhi)
{
  return FastL1RegionMap::getRegionTowerIndex(std::pair<int, int>(iEta,iPhi));
}

std::pair<int, int>
FastL1RegionMap::getRegionEtaPhiIndex(int regionId)
{
  int ieta = regionId%22;  
  int iphi = regionId/22;

  return std::pair<int, int>(ieta, iphi);
}

int
FastL1RegionMap::getRegionIndex(int ieta, int iphi)
{

  std::pair<int, int> ietaphi(ieta,iphi);
  std::pair<int, int> iep = getRegionEtaPhiIndex(ietaphi);

  int rgnid = iep.second*22 + iep.first;

  return rgnid;
}

int
FastL1RegionMap::getRegionIndex(CaloTowerDetId tower)
{
  return getRegionIndex(tower.ieta(), tower.iphi());
}

// ascii visualisation of mapping
void 
FastL1RegionMap::display() {
  // Region IDs
  for (int iRgn=0; iRgn<396; iRgn++) {
    if (iRgn%22 == 0) std::cerr << std::endl;   
    std::cerr << iRgn << " ";
  }

  for (int iRgn=0; iRgn<396; iRgn++) {
    if (iRgn%22 == 0) std::cerr << std::endl;   
    //std::pair<int, int> pep = m_Regions[iRgn].SetEtaPhiIndex();
    for (int iTwr=0; iTwr<16; iTwr++) {
      
      if (iTwr%4 == 0) std::cerr << " | ";   
      std::cerr << iRgn << " ";
    }
  }
  
}

std::pair<double, double>
FastL1RegionMap::getRegionCenterEtaPhi(int iRgn)
{

  std::pair<int, int> ep = getRegionEtaPhiIndex(iRgn);

  // this only true for barrel + endcap!
  double eta = 999.;  
  double phi = 999.;  

  // phi
  if (ep.second <= 9) {
    //phi = ep.second * 0.349065 + 0.1745329; // 10 degrees
    phi = ep.second * 0.349065 ;
    //phi = ep.second * 0.3490658504 + 0.1745329252; // 10 degrees
  } else {
    //phi = (18-ep.second)  * (-0.349065)  + 0.1745329; // -10 degrees
    phi = (18-ep.second)  * (-0.349065);
  }
  // eta
  if (ep.first >= 11 && ep.first <= 15 )
    eta = (ep.first-11)*0.349 + 0.1745;
    //eta = (ep.first-11)*0.3490658504 + 0.1745329252;
  if (ep.first == 16 )
    eta = 1.956;
  if (ep.first == 17 )
    eta = 2.586;
  if (ep.first == 18 )
    eta = 3.25;
  if (ep.first == 19 )
    eta = 3.75;
  if (ep.first == 20 )
    eta = 4.25;
  if (ep.first == 21 )
    eta = 4.75;

  if (ep.first >= 6 && ep.first <= 10 )
    eta = (10-ep.first)*(-0.348) - 0.174;
  if (ep.first == 5 )
    eta = -1.956;
  if (ep.first == 4 )
    eta = -2.586;
  if (ep.first == 3 )
    eta = -3.25;
  if (ep.first == 2 )
    eta = -3.75;
  if (ep.first == 1 )
    eta = -4.25;
  if (ep.first == 0 )
    eta = -4.75;

  //std::cout << "eta, phi ID: "<< ep.first << ", " << ep.second << std::endl;
  //std::cout << "eta, phi: "<< eta << ", " << phi << std::endl;
 

  return std::pair<double, double>(eta, phi);
}


// mapping from ECAL iphi numbering to HCAL iphi numbering
// which is shifted by 2 towers down
int
FastL1RegionMap::convertFromECal_to_HCal_iphi(int iphi_ecal)
{
  int iphi = 999;
  if (iphi_ecal>=3)
    iphi = iphi_ecal - 2;
  else if (iphi_ecal==1 || iphi_ecal==2)
    iphi = 70 + iphi_ecal;
 
  return iphi; 
}


// mapping from HCAL iphi numbering to ECAL iphi numbering
// which is shifted by 2 towers down
int
FastL1RegionMap::convertFromHCal_to_ECal_iphi(int iphi_hcal)
{
  int iphi = 999;
  if (iphi_hcal>=1 && iphi_hcal<=70)
    iphi = iphi_hcal + 2;
  else if (iphi_hcal==71 || iphi_hcal==72)
    iphi = iphi_hcal - 70;
 
  return iphi; 
}

std::pair<int, int> 
FastL1RegionMap::GetTowerNorthEtaPhi(int ieta, int iphi) 
{ 
  if (iphi < 72) 
    return std::pair<int, int>(ieta, iphi+1); 
  else 
    return std::pair<int, int>(ieta, 1); 
}

std::pair<int, int> 
FastL1RegionMap::GetTowerSouthEtaPhi(int ieta, int iphi) 
{ 
  if (iphi > 1) 
    return std::pair<int, int>(ieta, iphi-1); 
  else 
    return std::pair<int, int>(ieta, 72); 
}

std::pair<int, int> 
FastL1RegionMap::GetTowerWestEtaPhi(int ieta, int iphi) 
{ 
  if (ieta == 1) return std::pair<int, int>(-1, iphi);

  if (ieta > -32) 
    return std::pair<int, int>(ieta-1, iphi); 
  else 
    return std::pair<int, int>(999, iphi); 
}

std::pair<int, int> 
FastL1RegionMap::GetTowerEastEtaPhi(int ieta, int iphi) 
{ 
  if (ieta == -1) return std::pair<int, int>(1, iphi);

  if (ieta < 32) 
    return std::pair<int, int>(ieta+1, iphi); 
  else 
    return std::pair<int, int>(999, iphi); 
}

std::pair<int, int> 
FastL1RegionMap::GetTowerNWEtaPhi(int ieta, int iphi) 
{ 
  int iEta = ieta - 1;
  int iPhi = iphi + 1;
  if (ieta <= -32) 
    iEta = 999;
  if (ieta == 1) 
    iEta = -1;
  if (iphi == 72) 
    iPhi = 1;

  return std::pair<int, int>(iEta, iPhi); 
}

std::pair<int, int> 
FastL1RegionMap::GetTowerNEEtaPhi(int ieta, int iphi) 
{ 
  int iEta = ieta + 1;
  int iPhi = iphi + 1;
  if (ieta >= 32) 
    iEta = 999;
  if (ieta == -1) 
    iEta = 1;
  if (iphi == 72) 
    iPhi = 1;

  return std::pair<int, int>(iEta, iPhi); 
}



std::pair<int, int> 
FastL1RegionMap::GetTowerSWEtaPhi(int ieta, int iphi) 
{ 
  int iEta = ieta - 1;
  int iPhi = iphi - 1;
  if (ieta <= -32) 
    iEta = 999;
  if (ieta == 1) 
    iEta = -1;
  if (iphi == 1) 
    iPhi = 72;

  return std::pair<int, int>(iEta, iPhi); 
}

std::pair<int, int> 
FastL1RegionMap::GetTowerSEEtaPhi(int ieta, int iphi) 
{ 
  int iEta = ieta + 1;
  int iPhi = iphi - 1;
  if (ieta >= 32) 
    iEta = 999;
  if (ieta == -1) 
    iEta = 1;
  if (iphi == 1) 
    iPhi = 72;

  return std::pair<int, int>(iEta, iPhi); 
}
