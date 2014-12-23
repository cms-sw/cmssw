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
m_towers(0)
{
}

FWCandidateTowerProxyBuilder::~FWCandidateTowerProxyBuilder()
{
}

//
// member functions
//



void
FWCandidateTowerProxyBuilder::build(const FWEventItem* iItem, TEveElementList* el, const FWViewContext* ctx)
{
   m_towers=0;
   if (iItem)
   {
      iItem->get(m_towers);
      FWCaloDataProxyBuilderBase::build(iItem, el, ctx);
   }
}


FWHistSliceSelector*
FWCandidateTowerProxyBuilder::instantiateSliceSelector()
{
   FWCandidateTowerSliceSelector* ss = new FWCandidateTowerSliceSelector(m_hist, item());
   return ss;
}

void
FWCandidateTowerProxyBuilder::fillCaloData()
{
    m_hist->Reset();

    if (m_towers)
    {
        if(item()->defaultDisplayProperties().isVisible()) {
            // assert(item()->size() >= m_towers->size());
            unsigned int index=0;
            for( pat::PackedCandidateCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower,++index) {
                const FWEventItem::ModelInfo& info = item()->modelInfo(index);
                if(info.displayProperties().isVisible()) {
                   addEntryToTEveCaloData(tower->eta(), tower->phi(), getEt(*tower), info.isSelected());
                }
            }
        }
    }
}

double FWECalCandidateProxyBuilder::getEt(const reco::Candidate& iTower) const
{
      int  i = abs(iTower.pdgId());
      if (i == 22 || i == 13 || i == 111)
        return iTower.pt();
      else 
         return 0;

}


double FWHCalCandidateProxyBuilder::getEt(const reco::Candidate& iTower) const
{
      int  i = abs(iTower.pdgId());
      if (i == 211)
        return iTower.pt();
      else 
         return 0;
}

REGISTER_FWPROXYBUILDER(FWECalCandidateProxyBuilder, pat::PackedCandidateCollection,"CandECal",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kAllLegoBits);
REGISTER_FWPROXYBUILDER(FWHCalCandidateProxyBuilder, pat::PackedCandidateCollection,"CandHCal",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kAllLegoBits );
