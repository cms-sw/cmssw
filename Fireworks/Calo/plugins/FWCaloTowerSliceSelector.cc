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
// $Id: FWCaloTowerSliceSelector.cc,v 1.3 2010/12/01 21:40:31 amraktad Exp $
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
   FWFromSliceSelector(i),
   m_hist(h)
{
}

FWCaloTowerSliceSelector::~FWCaloTowerSliceSelector()
{
}

//
// member functions
//
//

bool
FWCaloTowerSliceSelector::matchCell(const TEveCaloData::CellId_t& iCell, const CaloTower& tower) const
{
   bool match = false;
   int idx = m_hist->FindBin(tower.eta(), tower.phi());
   int nBinsX = m_hist->GetXaxis()->GetNbins() + 2;

   int etaBin, phiBin, w, newPhiBin;
   m_hist->GetBinXYZ(idx, etaBin, phiBin, w);

   if (tower.ietaAbs() > 39)
   {
      newPhiBin =  ((phiBin + 1) / 4) * 4 - 1;
      if (newPhiBin <= 0) newPhiBin = 71;

      idx = etaBin + newPhiBin*nBinsX;
      match |= (idx == iCell.fTower);

      idx += nBinsX;
      match |= (idx == iCell.fTower);

      idx += nBinsX;
      if (newPhiBin == 71)
         idx = etaBin + 1*nBinsX;
      match |= (idx == iCell.fTower);

      idx += nBinsX;
      match |= (idx == iCell.fTower);
   } 
   else if (tower.ietaAbs() > 20)
   {
      newPhiBin =  ((phiBin  + 1) / 2) * 2 -1;
      idx = etaBin + newPhiBin*nBinsX;
      match = ( idx == iCell.fTower ||  idx + nBinsX == iCell.fTower);
   }
   else
   {
      match = ( idx == iCell.fTower);
   }
   return match;
}

void
FWCaloTowerSliceSelector::doSelect(const TEveCaloData::CellId_t& iCell)
{
   if (!m_item) return;

   const CaloTowerCollection* towers=0;
   m_item->get(towers);
   assert(0!=towers);
   int index = 0;

   FWChangeSentry sentry(*(m_item->changeManager()));
   for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower,++index)
   {
      if (m_item->modelInfo(index).m_displayProperties.isVisible() && !m_item->modelInfo(index).isSelected())
      {
         if (matchCell(iCell, *tower))
         {
            m_item->select(index);
            break;
         }
      }
   }
}

void
FWCaloTowerSliceSelector::doUnselect(const TEveCaloData::CellId_t& iCell)
{
   if (!m_item) return;
  
   //  std::cout <<"  doUnselect "<<std::endl;

   const CaloTowerCollection* towers=0;
   m_item->get(towers);
   assert(0!=towers);
   int index = 0;
   FWChangeSentry sentry(*(m_item->changeManager()));
   for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower,++index)
   {
      if ( m_item->modelInfo(index).m_displayProperties.isVisible() &&
           m_item->modelInfo(index).isSelected()) {
         if (matchCell(iCell, *tower))
         {
            //  std::cout <<"  doUnselect "<<index<<std::endl;
            m_item->unselect(index);
            break;
         }
      }
   }
}
