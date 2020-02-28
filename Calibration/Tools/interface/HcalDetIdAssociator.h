#ifndef HTrackAssociator_HHcalDetIdAssociator_h
#define HTrackAssociator_HHcalDetIdAssociator_h 1
// -*- C++ -*-
//
// Package:    HTrackAssociator
// Class:      HHcalDetIdAssociator
//
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>

//
// Original Author:  Michal Szleper
//

*/
//

#include "Calibration/Tools/interface/CaloDetIdAssociator.h"

class HHcalDetIdAssociator : public HCaloDetIdAssociator {
public:
  HHcalDetIdAssociator() : HCaloDetIdAssociator(72, 70, 0.087){};

protected:
  std::set<DetId> getASetOfValidDetIds() override {
    std::set<DetId> setOfValidIds;
    const std::vector<DetId>& vectOfValidIds = geometry_->getValidDetIds(DetId::Hcal, 1);  //HB
    for (std::vector<DetId>::const_iterator it = vectOfValidIds.begin(); it != vectOfValidIds.end(); ++it)
      setOfValidIds.insert(*it);

    //      vectOfValidIds.clear();
    const std::vector<DetId>& vectOfValidIdsHE = geometry_->getValidDetIds(DetId::Hcal, 2);  //HE
    for (std::vector<DetId>::const_iterator it = vectOfValidIdsHE.begin(); it != vectOfValidIdsHE.end(); ++it)
      setOfValidIds.insert(*it);

    return setOfValidIds;
  }
};
#endif
