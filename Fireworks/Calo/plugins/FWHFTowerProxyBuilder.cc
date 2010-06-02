// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWHFTowerProxyBuilder
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Mon May 31 16:41:27 CEST 2010
// $Id: FWHFTowerProxyBuilder.cc,v 1.2 2010/05/31 17:51:08 yanjuntu Exp $
//

// system include files

// user include files
#include "TEveCaloData.h"
#include "TEveCalo.h"
#include "TH2F.h"

#include "Fireworks/Calo/plugins/FWHFTowerProxyBuilder.h"
#include "Fireworks/Calo/plugins/FWHFTowerSliceSelector.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"


FWHFTowerProxyBuilder::FWHFTowerProxyBuilder():
   m_hits(0)
{
}


FWHFTowerProxyBuilder::~FWHFTowerProxyBuilder()
{
}

//
// member functions
//
void
FWHFTowerProxyBuilder::setCaloData(const fireworks::Context&)
{
  m_caloData = context().getCaloDataHF();
}

void
FWHFTowerProxyBuilder::addSliceSelector()
{
   FWFromTEveCaloDataSelector* sel = reinterpret_cast<FWFromTEveCaloDataSelector*>(context().getCaloData()->GetUserData());
   sel->addSliceSelector(m_sliceIndex, FWHFTowerSliceSelector(m_hist,item()));
}

void
FWHFTowerProxyBuilder::build(const FWEventItem* iItem,
                                  TEveElementList* el, const FWViewContext* ctx)
{
   m_hits=0;
   if (iItem) iItem->get(m_hits);
   FWCaloDataHistProxyBuilder::build(iItem, el, ctx);
}

//==============================================================================

void
FWHFShortTowerProxyBuilder::fillCaloData()
{
   m_hist->Reset();
   if (m_hits)
   {
      TEveCaloData::vCellId_t& selected = m_caloData->GetCellsSelected();

      if(item()->defaultDisplayProperties().isVisible()) {
         assert(item()->size() >= m_hits->size());
         unsigned int index=0;
         for(HFRecHitCollection::const_iterator it = m_hits->begin(); it != m_hits->end(); ++it,++index) {
            const FWEventItem::ModelInfo& info = item()->modelInfo(index);
            if(info.displayProperties().isVisible()) {
               Float_t energy = (*it).energy();
               std::vector<TEveVector> corners = item()->getGeom()->getPoints((*it).detid().rawId());
               if( corners.empty() ) {
                  break;
               }
               HcalDetId id ((*it).detid().rawId()); 
	       if(id.depth()==2)(m_hist)->Fill(corners.at(0).Eta(),corners.at(0).Phi(), energy*0.2); 

               if(info.isSelected())
               {
                  selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(corners.at(0).Eta(),corners.at(0).Phi()),m_sliceIndex));
               } 
            }
         }
      }
   }

}

void
FWHFLongTowerProxyBuilder::fillCaloData()
{ 
   m_hist->Reset();
   if (m_hits)
   {
      TEveCaloData::vCellId_t& selected = m_caloData->GetCellsSelected();

      if(item()->defaultDisplayProperties().isVisible()) {
         assert(item()->size() >= m_hits->size());
         unsigned int index=0;
         for(HFRecHitCollection::const_iterator it = m_hits->begin(); it != m_hits->end(); ++it,++index) {
            const FWEventItem::ModelInfo& info = item()->modelInfo(index);
            if(info.displayProperties().isVisible()) {
               Float_t energy = (*it).energy();
               std::vector<TEveVector> corners = item()->getGeom()->getPoints((*it).detid().rawId());
               if( corners.empty() ) {
                  break;
               }
               HcalDetId id ((*it).detid().rawId()); 
	       if(id.depth()==1)(m_hist)->Fill(corners.at(0).Eta(),corners.at(0).Phi(), energy*0.2); 

               if(info.isSelected())
               {
                  selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(corners.at(0).Eta(),corners.at(0).Phi()),m_sliceIndex));
               } 
            }
         }
      }
   }
}



REGISTER_FWPROXYBUILDER(FWHFShortTowerProxyBuilder, HFRecHitCollection, "HFShort", FWViewType::kLegoHFBit);
REGISTER_FWPROXYBUILDER(FWHFLongTowerProxyBuilder , HFRecHitCollection, "HFLong" , FWViewType::kLegoHFBit);


