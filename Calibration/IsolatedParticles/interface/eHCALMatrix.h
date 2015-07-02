/* 
Functions to return total energy contained in NxN (3x3/5x5/7x7)
Hcal towers aroud a given DetId. 

Inputs : 
1. HcalTopology, 
2. DetId around which NxN is to be formed, 
3. HcalRecHitCollection,
4. Number of towers to be navigated along eta and phi along 
   one direction (navigation is done alone +-deta and +-dphi).
5. option to include HO

Authors:  Seema Sharma, Sunanda Banerjee
Created: August 2009
*/

#ifndef CalibrationIsolatedParticleseHCALMatrix_h
#define CalibrationIsolatedParticleseHCALMatrix_h

// system include files
#include <memory>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

namespace spr{

  template< typename T>
  double eHCALmatrix(const HcalTopology* topology, const DetId& det, edm::Handle<T>& hits, int ieta, int iphi, bool includeHO=false, bool algoNew=true, double hbThr=-100, double heThr=-100, double hfThr=-100, double hoThr=-100, double tMin=-500, double tMax=500, bool debug=false);

  template< typename T>
  double eHCALmatrix(const HcalTopology* topology, const DetId& det, edm::Handle<T>& hits, int ietaE, int ietaW, int iphiN, int iphiS, bool includeHO=false, double hbThr=-100, double heThr=-100, double hfThr=-100, double hoThr=-100, double tMin=-500, double tMax=500, bool debug=false);

  template <typename T>
  double eHCALmatrix(const CaloGeometry* geo, const HcalTopology* topology, const DetId& det0, edm::Handle<T>& hits, int ieta, int iphi, int& nRecHits, std::vector<int>& RH_ieta, std::vector<int>& RH_iphi, std::vector<double>& RH_ene, GlobalPoint& gPosHotCell);

  template< typename T>
  double eHCALmatrix(const HcalTopology* topology, const DetId& det0, edm::Handle<T>& hits, int ieta, int iphi, int& nRecHits, std::vector<int>& RH_ieta, std::vector<int>& RH_iphi, std::vector<double>& RH_ene, std::set<int>& uniqueIdset);

  template< typename T>
  double eHCALmatrix(const HcalTopology* topology, const DetId& det0, edm::Handle<T>& hits, int ieta, int iphi, HcalDetId& hotCell, bool includeHO=false, bool debug=false);

  template <typename T>
  double energyHCALmatrixNew(const HcalTopology* topology, const DetId& det, edm::Handle<T>& hits, int ieta, int iphi, bool includeHO=false, double hbThr=-100, double heThr=-100, double hfThr=-100, double hoThr=-100, double tMin=-500, double tMax=500, bool debug=false);

  template <typename T>
  double energyHCALmatrixTotal(const HcalTopology* topology, const DetId& det, edm::Handle<T>& hits, int ietaE, int ietaW, int iphiN, int iphiS, bool includeHO=false, double hbThr=-100, double heThr=-100, double hfThr=-100, double hoThr=-100, double tMin=-500, double tMax=500, bool debug=false);

  template <typename T>
  void hitHCALmatrix(const HcalTopology* topology, const DetId& det, edm::Handle<T>& hits, int ieta, int iphi, std::vector< typename T::const_iterator>& hitlist,  bool includeHO=false, bool debug=false);

  template <typename T>
  void hitHCALmatrixTotal(const HcalTopology* topology, const DetId& det, edm::Handle<T>& hits, int ietaE, int ietaW, int iphiN, int iphiS, std::vector< typename T::const_iterator>& hitlist, bool includeHO=false, bool debug=false);
 
  template <typename T>
  double energyHCAL(std::vector<DetId>& vdets, edm::Handle<T>& hits, double hbThr=-100, double heThr=-100, double hfThr=-100, double hoThr=-100, double tMin=-500, double tMax=500, bool debug=false);
 
  template <typename T>
  void energyHCALCell(HcalDetId detId, edm::Handle<T>& hits, std::vector<std::pair<double,int> >& energyCell, int maxDepth=1, double hbThr=-100, double heThr=-100, double hfThr=-100, double hoThr=-100, double tMin=-500, double tMax=500, bool debug=false);

  template <typename T>
  void hitsHCAL(std::vector<DetId>& vdets, edm::Handle<T>& hits, std::vector< typename T::const_iterator>& hitlist, bool debug=false);

  HcalDetId getHotCell(std::vector<HBHERecHitCollection::const_iterator>& hit, bool& includeHO, bool& debug);

  HcalDetId getHotCell(std::vector<std::vector<PCaloHit>::const_iterator>& hit, bool& includeHO, bool& debug);
}

#include "Calibration/IsolatedParticles/interface/eHCALMatrix.icc"
#endif
