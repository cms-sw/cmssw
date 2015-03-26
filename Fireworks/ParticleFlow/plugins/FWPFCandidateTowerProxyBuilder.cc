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
#include "Fireworks/ParticleFlow/plugins/FWPFCandidateTowerProxyBuilder.h"
#include "Fireworks/ParticleFlow/plugins/FWPFCandidateTowerSliceSelector.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"



//
// constructors , dectructors
//
FWPFCandidateTowerProxyBuilder::FWPFCandidateTowerProxyBuilder():
m_towers(0)
{
}

FWPFCandidateTowerProxyBuilder::~FWPFCandidateTowerProxyBuilder()
{
}

//
// member functions
//



void
FWPFCandidateTowerProxyBuilder::build(const FWEventItem* iItem, TEveElementList* el, const FWViewContext* ctx)
{
   m_towers=0;
   if (iItem)
   {
      iItem->get(m_towers);
      FWCaloDataProxyBuilderBase::build(iItem, el, ctx);
   }
}


FWHistSliceSelector*
FWPFCandidateTowerProxyBuilder::instantiateSliceSelector()
{
   return new FWPFCandidateTowerSliceSelector(m_hist, item());
}

void
FWPFCandidateTowerProxyBuilder::fillCaloData()
{
    m_hist->Reset();

    if (m_towers)
    {
        if(item()->defaultDisplayProperties().isVisible()) {
            // assert(item()->size() >= m_towers->size());
            unsigned int index=0;
            for( reco::PFCandidateConstIterator tower = m_towers->begin(); tower != m_towers->end(); ++tower,++index) {
                const FWEventItem::ModelInfo& info = item()->modelInfo(index);
                if(info.displayProperties().isVisible()) {
                    addEntryToTEveCaloData(tower->eta(), tower->phi(), getEt(*tower), info.isSelected());
                }
            }
        }
    }
}

REGISTER_FWPROXYBUILDER(FWECalPFCandidateProxyBuilder, reco::PFCandidateCollection,"CaloTowerPfCandEcal",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kAllLegoBits);
REGISTER_FWPROXYBUILDER(FWHCalPFCandidateProxyBuilder, reco::PFCandidateCollection,"CaloTowerPfCandHcal",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kAllLegoBits);
