#ifndef Fireworks_Calo_FWHGTowerSliceSelector_h
#define Fireworks_Calo_FWHGTowerSliceSelector_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWHGTowerSliceSelector
// 
/**\class FWHGTowerSliceSelector FWHGTowerSliceSelector.h Fireworks/Calo/interface/FWHGTowerSliceSelector.h

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
class DetId;
class TEveCaloDataVec;

#include "Fireworks/Calo/src/FWFromSliceSelector.h"

// forward declarations

class FWHGTowerSliceSelector : public FWFromSliceSelector
{
public:
   FWHGTowerSliceSelector(const FWEventItem* i, TEveCaloDataVec* data) : 
      FWFromSliceSelector(i), m_vecData(data) {}

   ~FWHGTowerSliceSelector() override {}

   void doSelect(const TEveCaloData::CellId_t&) override;
   void doUnselect(const TEveCaloData::CellId_t&) override;
   
private:
   bool findBinFromId(DetId& id, int tower) const;
   TEveCaloDataVec* m_vecData;
};


#endif
