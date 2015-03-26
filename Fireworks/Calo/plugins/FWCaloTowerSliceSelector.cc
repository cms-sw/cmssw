// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloTowerSliceSelector
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Wed Jun  2 17:36:23 CEST 2010
//

// system include files

// user include files
#include "TH2F.h"
#include "TMath.h"
#include "Fireworks/Calo/plugins/FWCaloTowerSliceSelector.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"


FWCaloTowerSliceSelector::FWCaloTowerSliceSelector(TH2F* h, const FWEventItem* i):
   FWHistSliceSelector(h, i)
{
}

FWCaloTowerSliceSelector::~FWCaloTowerSliceSelector()
{
}

void
FWCaloTowerSliceSelector::getItemEntryEtaPhi(int itemIdx, float& eta, float& phi) const
{
    const CaloTowerCollection* towers=0;
    m_item->get(towers);
    assert(0!=towers);
    CaloTowerCollection::const_iterator tower = towers->begin();
    std::advance(tower, itemIdx);

    eta = tower->eta();
    phi = tower->phi();
}
