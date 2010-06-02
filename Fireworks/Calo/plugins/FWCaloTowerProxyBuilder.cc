// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloTowerProxyBuilderBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Dec  3 11:28:28 EST 2008
// $Id: FWCaloTowerProxyBuilder.cc,v 1.15 2010/06/02 19:08:33 amraktad Exp $
//

// system includes
#include <math.h>

// user includes
#include "TEveCaloData.h"
#include "TEveCalo.h"
#include "TH2F.h"

#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"

#include "Fireworks/Calo/plugins/FWCaloTowerProxyBuilder.h"
#include "Fireworks/Calo/plugins/FWCaloTowerSliceSelector.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"



//
// constructors , dectructors
//
FWCaloTowerProxyBuilderBase::FWCaloTowerProxyBuilderBase():
   m_towers(0)
{
}

FWCaloTowerProxyBuilderBase::~FWCaloTowerProxyBuilderBase()
{
}

//
// member functions
//

void
FWCaloTowerProxyBuilderBase::setCaloData(const fireworks::Context&)
{
  m_caloData = context().getCaloData();
}

void
FWCaloTowerProxyBuilderBase::addSliceSelector()
{
   FWFromTEveCaloDataSelector* sel = reinterpret_cast<FWFromTEveCaloDataSelector*>(m_caloData->GetUserData());
   sel->addSliceSelector(m_sliceIndex, new FWCaloTowerSliceSelector(m_hist,item()));
}

void
FWCaloTowerProxyBuilderBase::build(const FWEventItem* iItem,
                                  TEveElementList* el, const FWViewContext* ctx)
{
   m_towers=0;
   if (iItem) iItem->get(m_towers);
   FWCaloDataHistProxyBuilder::build(iItem, el, ctx);
}


void
FWCaloTowerProxyBuilderBase::fillCaloData()
{
   m_hist->Reset();

   if (m_towers)
   {
      TEveCaloData::vCellId_t& selected = m_caloData->GetCellsSelected();

      if(item()->defaultDisplayProperties().isVisible()) {
         // assert(item()->size() >= m_towers->size());
         unsigned int index=0;
         for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower,++index) {
            const FWEventItem::ModelInfo& info = item()->modelInfo(index);
            if(info.displayProperties().isVisible()) {
               (m_hist)->Fill(tower->eta(),tower->phi(), getEt(*tower));
            }
            if(info.isSelected()) {
               //NOTE: I tried calling TEveCalo::GetCellList but it always returned 0, probably because of threshold issues
               // but looking at the TEveCaloHist::GetCellList code the CellId_t is just the histograms bin # and the slice
               // printf("applyChangesToAllModels ...check selected \n");
               selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(tower->eta(),tower->phi()),m_sliceIndex));
            }
         }
      }
   }

}

bool
FWCaloTowerProxyBuilderBase::representsSubPart()
{
   return true;
}

REGISTER_FWPROXYBUILDER(FWECalCaloTowerProxyBuilder,CaloTowerCollection,"ECal",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kLegoBit);
REGISTER_FWPROXYBUILDER(FWHCalCaloTowerProxyBuilder,CaloTowerCollection,"HCal",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kLegoBit );

