#ifndef Fireworks_Calo_FWHFTowerSliceSelector_h
#define Fireworks_Calo_FWHFTowerSliceSelector_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWHFTowerSliceSelector
//
/**\class FWHFTowerSliceSelector FWHFTowerSliceSelector.h Fireworks/Calo/interface/FWHFTowerSliceSelector.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Wed Jun  2 19:21:13 CEST 2010
//

// system include files

// user include files
class HcalDetId;
class TEveCaloDataVec;

#include "Fireworks/Calo/interface/FWFromSliceSelector.h"

// forward declarations

class FWHFTowerSliceSelector : public FWFromSliceSelector {
public:
  FWHFTowerSliceSelector(const FWEventItem* i, TEveCaloDataVec* data) : FWFromSliceSelector(i), m_vecData(data) {}

  ~FWHFTowerSliceSelector() override {}

  void doSelect(const TEveCaloData::CellId_t&) override;
  void doUnselect(const TEveCaloData::CellId_t&) override;

private:
  bool findBinFromId(HcalDetId& id, int tower) const;
  TEveCaloDataVec* m_vecData;
};

#endif
