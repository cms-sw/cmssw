#include "Fireworks/Candidates/plugins/FWCandidateTowerProxyBuilder.h"

// system includes
#include <math.h>

// user includes
#include "TEveCaloData.h"
#include "TEveCalo.h"
#include "TH2F.h"

#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Candidates/interface/FWCandidateTowerSliceSelector.h"

#include "Fireworks/Core/interface/fw3dlego_xbins.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"



//
// constructors , dectructors
//
FWCandidateTowerProxyBuilder::FWCandidateTowerProxyBuilder():
   m_helper(typeid(reco::Candidate))
{
}

FWCandidateTowerProxyBuilder::~FWCandidateTowerProxyBuilder()
{
}

//
// member functions
//


void FWCandidateTowerProxyBuilder::itemChangedImp(const FWEventItem* iItem)
{
   if (iItem)
   {
      m_helper.itemChanged(iItem);
   }
}

void
FWCandidateTowerProxyBuilder::build(const FWEventItem* iItem, TEveElementList* el, const FWViewContext* ctx)
{
   //  m_towers=0;
   if (iItem )
   {
      //      iItem->get(m_towers);
      FWCaloDataProxyBuilderBase::build(iItem, el, ctx);
   }
}


FWHistSliceSelector*
FWCandidateTowerProxyBuilder::instantiateSliceSelector()
{
   FWCandidateTowerSliceSelector* ss = new FWCandidateTowerSliceSelector(m_hist, item(), &m_helper);
   return ss;
}

void
FWCandidateTowerProxyBuilder::fillCaloData()
{
    m_hist->Reset();

    //  if (m_towers)
    {
        if(item()->defaultDisplayProperties().isVisible()) {
            // assert(item()->size() >= m_towers->size());
           for (size_t index = 0; index < item()->size(); ++index) {
                const FWEventItem::ModelInfo& info = item()->modelInfo(index);
                if(info.displayProperties().isVisible()) {
                   const void* modelData = item()->modelData((int)index);

                   const reco::Candidate* tower = reinterpret_cast<const reco::Candidate*>(m_helper.offsetObject(modelData));
                   addEntryToTEveCaloData(tower->eta(), tower->phi(), getEt(*tower), info.isSelected());
                }
            }
        }
    }
}




REGISTER_FWPROXYBUILDER(FWCandidateTowerProxyBuilder, reco::Candidate,"CaloTower",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kAllLegoBits);
