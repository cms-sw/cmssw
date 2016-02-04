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
// $Id: FWHFTowerSliceSelector.cc,v 1.9 2010/09/07 15:46:45 yana Exp $
//

// system include files

// user include files
#include "TEveVector.h"
#include "TEveCaloData.h"
#include "TH2F.h"

#include "Fireworks/Calo/plugins/FWHFTowerSliceSelector.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
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
      HcalDetId id ((*it).detid().rawId());
      if (findBinFromId(id, iCell.fTower) && 
          m_item->modelInfo(index).m_displayProperties.isVisible() &&
          !m_item->modelInfo(index).isSelected()) {
         // std::cout <<"  doSelect "<<index<<std::endl;
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
      HcalDetId id ((*it).detid().rawId());
      if (findBinFromId(id, iCell.fTower) && 
          m_item->modelInfo(index).m_displayProperties.isVisible() &&
          m_item->modelInfo(index).isSelected()) {
         // std::cout <<"  doUnselect "<<index<<std::endl;
         m_item->unselect(index);
      }
   }
}

bool
FWHFTowerSliceSelector::findBinFromId( HcalDetId& detId, int tower) const
{    
   TEveCaloData::vCellId_t cellIds;
   const float* corners = m_item->getGeom()->getCorners( detId.rawId());
   if( corners == 0 )
   {
     fwLog( fwlog::kInfo ) << "FWHFTowerSliceSelector cannot get geometry for DetId: "<< detId.rawId() << ". Ignored.\n";
     return false;
   }
   std::vector<TEveVector> front( 4 );
   float eta = 0, phi = 0;
   int j = 0;
   for( int i = 0; i < 4; ++i )
   {
     front[i] = TEveVector( corners[j], corners[j + 1], corners[j + 2] );
     j +=3;

     eta += front[i].Eta();
     phi += front[i].Phi();
   }
   eta /= 4;
   phi /= 4;

   const TEveCaloData::CellGeom_t &cg = m_vecData->GetCellGeom()[tower] ;
   if(( eta >= cg.fEtaMin && eta <= cg.fEtaMax) && (phi >= cg.fPhiMin && phi <= cg.fPhiMax))
   {
      return true;
   }

   return false;
}
