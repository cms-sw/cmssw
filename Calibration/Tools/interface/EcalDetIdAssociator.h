#ifndef HTrackAssociator_HEcalDetIdAssociator_h
#define HTrackAssociator_HEcalDetIdAssociator_h 1
// -*- C++ -*-
//
// Package:    HTrackAssociator
// Class:      HEcalDetIdAssociator
//
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
// Modified for ECAL+HCAL by Michal Szleper
//

#include "Calibration/Tools/interface/CaloDetIdAssociator.h"

class HEcalDetIdAssociator : public HCaloDetIdAssociator {
public:
  HEcalDetIdAssociator() : HCaloDetIdAssociator(180, 150, 0.04){};

protected:
  std::set<DetId> getASetOfValidDetIds() override {
    std::set<DetId> setOfValidIds;
    const std::vector<DetId>& vectOfValidIds = geometry_->getValidDetIds(DetId::Ecal, 1);  //EB
    for (auto vectOfValidId : vectOfValidIds)
      setOfValidIds.insert(vectOfValidId);

    //      vectOfValidIds.clear();
    const std::vector<DetId>& vectOfValidIdsEE = geometry_->getValidDetIds(DetId::Ecal, 2);  //EE
    for (auto it : vectOfValidIdsEE)
      setOfValidIds.insert(it);

    return setOfValidIds;
  }
};
#endif
