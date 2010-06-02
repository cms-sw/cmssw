// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWHFTowerSliceSelector
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Wed Jun  2 17:39:44 CEST 2010
// $Id$
//

// system include files

// user include files
#include "TEveVector.h"
#include "TH2F.h"

#include "Fireworks/Calo/plugins/FWHFTowerSliceSelector.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"




//
// member functions
//

void
FWHFTowerSliceSelector::doSelect(const TEveCaloData::CellId_t& iCell)
{
   if (!m_item) return;

   const HFRecHitCollection* hits=0;
   m_item->get(hits);
   assert(0!=hits);

   int index = 0;
   FWChangeSentry sentry(*(m_item->changeManager()));
   for(HFRecHitCollection::const_iterator it = hits->begin(); it != hits->end(); ++it,++index)
   {
      std::vector<TEveVector> corners = m_item->getGeom()->getPoints((*it).detid().rawId());
      if (m_hist->FindBin(corners.at(0).Eta(),corners.at(0).Phi()) == iCell.fTower && 
          m_item->modelInfo(index).m_displayProperties.isVisible() &&
          !m_item->modelInfo(index).isSelected()) {
         //std::cout <<"  doSelect "<<index<<std::endl;
         m_item->select(index);
      }
   }
}

void
FWHFTowerSliceSelector::doUnselect(const TEveCaloData::CellId_t& iCell)
{
   if (!m_item) return;

   const HFRecHitCollection* hits=0;
   m_item->get(hits);
   assert(0!=hits);

   int index = 0;
   FWChangeSentry sentry(*(m_item->changeManager()));
   for(HFRecHitCollection::const_iterator it = hits->begin(); it != hits->end(); ++it,++index)
   {
      std::vector<TEveVector> corners = m_item->getGeom()->getPoints((*it).detid().rawId());
      if (m_hist->FindBin(corners.at(0).Eta(),corners.at(0).Phi())  == iCell.fTower && 
          m_item->modelInfo(index).m_displayProperties.isVisible() &&
          m_item->modelInfo(index).isSelected()) {
         //std::cout <<"  doUnselect "<<index<<std::endl;
         m_item->unselect(index);
      }
   }
}

