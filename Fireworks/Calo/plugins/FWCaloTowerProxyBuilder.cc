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

#include "Fireworks/Core/interface/fw3dlego_xbins.h"
#include "Fireworks/Calo/plugins/FWCaloTowerProxyBuilder.h"
#include "Fireworks/Calo/plugins/FWCaloTowerSliceSelector.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"



//
// constructors , dectructors
//
FWCaloTowerProxyBuilderBase::FWCaloTowerProxyBuilderBase():
   FWCaloDataHistProxyBuilder(),
   m_towers(0)
{
}

FWCaloTowerProxyBuilderBase::~FWCaloTowerProxyBuilderBase()
{
}


void
FWCaloTowerProxyBuilderBase::build(const FWEventItem* iItem,
                                   TEveElementList* el, const FWViewContext* ctx)
{
   m_towers=0;
   if (iItem)
   {
      iItem->get(m_towers);
      FWCaloDataProxyBuilderBase::build(iItem, el, ctx);
   }
}


FWHistSliceSelector*
FWCaloTowerProxyBuilderBase::instantiateSliceSelector()
{
    return new FWCaloTowerSliceSelector(m_hist, item());
}


void
FWCaloTowerProxyBuilderBase::fillCaloData()
{
   m_hist->Reset();

   if (m_towers)
   {
      if(item()->defaultDisplayProperties().isVisible()) {
         unsigned int index=0;
         for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower,++index) {
            const FWEventItem::ModelInfo& info = item()->modelInfo(index);
            if(info.displayProperties().isVisible()) {
               addEntryToTEveCaloData(tower->eta(), tower->phi(), getEt(*tower), info.isSelected());
            }
         }
      }
   }
}



REGISTER_FWPROXYBUILDER(FWECalCaloTowerProxyBuilder,CaloTowerCollection,"ECal",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kAllLegoBits);
REGISTER_FWPROXYBUILDER(FWHCalCaloTowerProxyBuilder,CaloTowerCollection,"HCal",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kAllLegoBits );
REGISTER_FWPROXYBUILDER(FWHOCaloTowerProxyBuilder,CaloTowerCollection,"HCal Outer",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kAllLegoBits );



