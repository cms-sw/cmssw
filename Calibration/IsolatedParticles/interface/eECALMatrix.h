// -*- C++ -*
/* 
The function eECALmatrix returns total energy contained in 
NxN crystal matrix for EcalRecHits or PCaloSimHits.

Inputs : 
1. CaloNavigator at the DetId around which NxN has to be formed
2. The EcalRecHitCollection  and 
3. Number of crystals to be navigated along eta and phi along 
   one direction (navigation is done alone +-deta and +-dphi).

Authors:  Seema Sharma, Sunanda Banerjee
Created: August 2009
*/


#ifndef CalibrationIsolatedParticleseECALMatrix_h
#define CalibrationIsolatedParticleseECALMatrix_h

// system include files
#include <memory>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"


#include "Calibration/IsolatedParticles/interface/MatrixECALDetIds.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

class EcalSeverityLevelAlgo;

namespace spr{

  // Energy in NxN crystal matrix
  template< typename T>
  double eECALmatrix(const DetId& detId, edm::Handle<T>& hitsEB, edm::Handle<T>& hitsEE, const CaloGeometry* geo, const CaloTopology* caloTopology, int ieta, int iphi, double ebThr=-100, double eeThr=-100, double tMin=-500, double tMax=500,  bool debug=false);

  template< typename T>
  double eECALmatrix(const DetId& detId, edm::Handle<T>& hitsEB, edm::Handle<T>& hitsEE, const CaloGeometry* geo, const CaloTopology* caloTopology, const EcalTrigTowerConstituentsMap& ttMap, int ieta, int iphi, double ebThr=-100, double eeThr=-100, double tMin=-500, double tMax=500, bool debug=false);

  template< typename T>
  double eECALmatrix(const DetId& detId, edm::Handle<T>& hitsEB, edm::Handle<T>& hitsEE, const CaloGeometry* geo, const CaloTopology* caloTopology, int ietaE, int ietaW, int iphiN, int iphiS, double ebThr=-100, double eeThr=-100, double tMin=-500, double tMax=500,bool debug=false);

  std::pair <double,bool> eECALmatrix(const DetId& detId, edm::Handle<EcalRecHitCollection>& hitsEB, edm::Handle<EcalRecHitCollection>& hitsEE, const EcalChannelStatus& chStatus, const CaloGeometry* geo, const CaloTopology* caloTopology, const EcalSeverityLevelAlgo* sevlv,int ieta, int iphi, double ebThr=-100, double eeThr=-100, double tMin=-500, double tMax=500,  bool debug=false);

  std::pair <double,bool> eECALmatrix(const DetId& detId, edm::Handle<EcalRecHitCollection>& hitsEB, edm::Handle<EcalRecHitCollection>& hitsEE, const EcalChannelStatus& chStatus, const CaloGeometry* geo, const CaloTopology* caloTopology, const EcalSeverityLevelAlgo* sevlv,const EcalTrigTowerConstituentsMap& ttMap, int ieta, int iphi, double ebThr=-100, double eeThr=-100, double tMin=-500, double tMax=500, bool debug=false);
  
  // returns vector of hits in NxN matrix 
  template <typename T>
  void hitECALmatrix(CaloNavigator<DetId>& navigator, edm::Handle<T>& hits, int ieta, int iphi, std::vector<typename T::const_iterator>& hitlist, bool debug=false);
  
  // returns energy deposited from the vector of hits
  template <typename T>
  double energyECAL(std::vector<DetId>& vdets, edm::Handle<T>& hitsEB,  edm::Handle<T>& hitsEE, double ebThr=-100, double eeThr=-100, double tMin=-500, double tMax=500, bool debug=false);

  template <typename T>
  double energyECAL(std::vector<DetId>& vdets, edm::Handle<T>& hitsEB,  edm::Handle<T>& hitsEE, const EcalTrigTowerConstituentsMap& ttMap, double ebThr=-100, double eeThr=-100, double tMin=-500, double tMax=500, bool debug=false);

  // returns energy in the EB/EE tower 
  template <typename T>
  double energyECALTower(const DetId& detId, edm::Handle<T>& hitsEB, edm::Handle<T>& hitsEE, const EcalTrigTowerConstituentsMap& ttMap, bool debug=false);

  // Hot Crystal
  template< typename T>
  DetId hotCrystal(const DetId& detId, edm::Handle<T>& hitsEB, edm::Handle<T>& hitsEE, const CaloGeometry* geo, const CaloTopology* caloTopology, int ieta, int iphi, double tMin=-500, double tMax=500,  bool debug=false);

  template< typename T>
  DetId hotCrystal(std::vector<DetId>& detId, edm::Handle<T>& hitsEB, edm::Handle<T>& hitsEE, double tMin=-500, double tMax=500,  bool debug=false);
}

#include "Calibration/IsolatedParticles/interface/eECALMatrix.icc"

#endif
