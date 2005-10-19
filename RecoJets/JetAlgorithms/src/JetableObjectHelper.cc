#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "RecoJets/JetAlgorithms/interface/JetableObjectHelper.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;

// JetableObjectHelper.cc: Methods used by JetableObjectHelper
// Author:  Robert M Harris
// History: 2/28/05 RMH, Initial Version (called CaloTowerHelper) for EDM Demo.  No attempt at optimization.
//          3/24/05 RMH, Added the eta-phi CaloTowerGrid and the methods to work with it. 
//          4/20/05 RMH, Added getNearestTower to find the ieta and iphi corresponding to eta and phi for ORCA 8 geometry.
//         10/19/05 RMH, Stipped out CaloTowerGrid, made work with new CaloTowers, and renamed it JetableObjectHelper.
//
// towersWithinCone returns a list of pointers to CaloTowers with Et>etThreshold within coneRadius
// in eta-phi space of the coneEta and conePhi.
std::vector<const CaloTower*> JetableObjectHelper::towersWithinCone(double coneEta, double conePhi, double coneRadius, double etThreshold){
  std::vector<const CaloTower *> result;
  CaloTowerCollection::const_iterator towerIter = caloTowerCollPointer->begin();
  CaloTowerCollection::const_iterator towerIterEnd = caloTowerCollPointer->end();
  for (;towerIter != towerIterEnd; ++towerIter) {
    const CaloTower *caloTowerPointer = &*towerIter;
    if(caloTowerPointer->eT > etThreshold){
      double towerEta = caloTowerPointer->eta;
      double towerPhi = caloTowerPointer->phi;
      double deltaEta = towerEta - coneEta;
      double deltaPhi = phidif(towerPhi, conePhi);
      double deltaR = sqrt(deltaEta*deltaEta + deltaPhi*deltaPhi);
      if(deltaR < coneRadius){
        if(caloTowerPointer != 0) result.push_back(caloTowerPointer);
      }
    }
  }
  return result;
}

// GreaterByET is used to sort by et
class GreaterByET 
{
public:
  bool operator()(const CaloTower * a, 
                   const CaloTower * b) const {
    return a->eT > b->eT;
  }
};


// etOrderedCaloTowers returns an Et order list of pointers to CaloTowers with Et>etTreshold
std::vector<const CaloTower*> JetableObjectHelper::etOrderedCaloTowers(double etThreshold) const {
  std::vector<const CaloTower*> result;
  CaloTowerCollection::const_iterator towerIter = caloTowerCollPointer->begin();
  CaloTowerCollection::const_iterator towerIterEnd = caloTowerCollPointer->end();
  for (;towerIter != towerIterEnd; ++towerIter) {
    const CaloTower *caloTowerPointer = &*towerIter;
    if(caloTowerPointer->eT > etThreshold){
      if(caloTowerPointer != 0) result.push_back(caloTowerPointer);
    }
  }   
  sort(result.begin(), result.end(), GreaterByET());
  return result;
}



 // phidif calculates the difference between phi1 and phi2 taking into account the 2pi issue.
double JetableObjectHelper::phidif(double phi1, double phi2) {
  double dphi = phi1 - phi2;
  if(dphi > M_PI) dphi -= 2*M_PI;
  if(dphi < -1*M_PI) dphi += 2*M_PI;
  return fabs(dphi);
}
