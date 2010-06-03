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
// $Id: FWHFTowerSliceSelector.cc,v 1.2 2010/06/02 19:08:33 amraktad Exp $
//

// system include files

// user include files
#include "TEveVector.h"
#include "TH2F.h"

#include "Fireworks/Calo/plugins/FWHFTowerSliceSelector.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"
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

      HcalDetId id ((*it).detid().rawId());
      if (findBinFromId(id, iCell.fTower) && 
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

      HcalDetId id ((*it).detid().rawId());
      if (findBinFromId(id, iCell.fTower) && 
          m_item->modelInfo(index).m_displayProperties.isVisible() &&
          m_item->modelInfo(index).isSelected()) {
         //std::cout <<"  doUnselect "<<index<<std::endl;
         m_item->unselect(index);
      }
   }
}

bool
FWHFTowerSliceSelector::findBinFromId( HcalDetId& id, int tower) const
{
   std::vector<TEveVector> corners = m_item->getGeom()->getPoints(id);
   TEveVector centre = corners[0] + corners[1] + corners[2] + corners[3] + corners[4] + corners[5] + corners[6] + corners[7];
   centre *= 1.0f / 8.0f;
   int bin = m_hist->FindBin(centre.Eta(),centre.Phi());
   if (bin == -1)
   {
      fwLog(fwlog::kWarning) << "FWHFTowerSliceSelector could not find a valid bin for eta: " << centre.Eta() << " phi: " << centre.Phi() << std::endl;
      fflush(stdout);
   }
   if (m_hist->FindBin(centre.Eta(),centre.Phi()) == tower)
   {
      fwLog(fwlog::kDebug) << "Secondary Selection "<< m_hist->GetName() <<"::  FWHFTowerSliceSelector find detId for tower: " << centre.Eta() << " phi: " << centre.Phi() << "val: " << m_hist->GetBinContent(bin) << std::endl;
      return true;
   }
   return false;
}
