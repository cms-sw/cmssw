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
// $Id$
//

// system include files

// user include files
#include "TH2F.h"
#include "Fireworks/Calo/plugins/FWCaloTowerSliceSelector.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"



//
// member functions
//
//

void
FWCaloTowerSliceSelector::doSelect(const TEveCaloData::CellId_t& iCell)
{
   if (!m_item) return;

   const CaloTowerCollection* towers=0;
   m_item->get(towers);
   assert(0!=towers);
   int index = 0;
   FWChangeSentry sentry(*(m_item->changeManager()));
   for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower,++index) {
      if (m_hist->FindBin(tower->eta(),tower->phi()) == iCell.fTower && 
          m_item->modelInfo(index).m_displayProperties.isVisible() &&
          !m_item->modelInfo(index).isSelected()) {
         //std::cout <<"  doSelect "<<index<<std::endl;
         m_item->select(index);
      }
   }
}

void
FWCaloTowerSliceSelector::doUnselect(const TEveCaloData::CellId_t& iCell)
{
   if (!m_item) return;

   const CaloTowerCollection* towers=0;
   m_item->get(towers);
   assert(0!=towers);
   int index = 0;
   FWChangeSentry sentry(*(m_item->changeManager()));
   for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower,++index) {
      if (m_hist->FindBin(tower->eta(),tower->phi()) == iCell.fTower && 
          m_item->modelInfo(index).m_displayProperties.isVisible() &&
          m_item->modelInfo(index).isSelected()) {
         //std::cout <<"  doUnselect "<<index<<std::endl;
         m_item->unselect(index);
      }
   }
}
