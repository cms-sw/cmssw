#ifndef Fireworks_Calo_FWCaloTowerSliceSelector_h
#define Fireworks_Calo_FWCaloTowerSliceSelector_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloTowerSliceSelector
// 
/**\class FWCaloTowerSliceSelector FWCaloTowerSliceSelector.h Fireworks/Calo/interface/FWCaloTowerSliceSelector.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Wed Jun  2 19:21:19 CEST 2010
//

// system include files

// user include files

#include "Fireworks/Calo/interface/FWHistSliceSelector.h"


class FWCaloTowerSliceSelector : public FWHistSliceSelector
{
public:
  FWCaloTowerSliceSelector(TH2F* h, const FWEventItem* i);
  virtual ~FWCaloTowerSliceSelector();
 
protected:
   virtual void getItemEntryEtaPhi(int itemIdx, float& eta, float& phi) const; 
};

#endif
