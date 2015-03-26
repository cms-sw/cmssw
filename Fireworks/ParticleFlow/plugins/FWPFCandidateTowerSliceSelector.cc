#include "Fireworks/ParticleFlow/plugins/FWPFCandidateTowerSliceSelector.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"


FWPFCandidateTowerSliceSelector::FWPFCandidateTowerSliceSelector(TH2F* h, const FWEventItem* i):
   FWHistSliceSelector(h, i)
{
}


FWPFCandidateTowerSliceSelector::~FWPFCandidateTowerSliceSelector()
{
}
void
FWPFCandidateTowerSliceSelector::getItemEntryEtaPhi(int itemIdx, float& eta, float& phi) const
{
   
   const reco::PFCandidateCollection* towers=0;
   m_item->get(towers);
   assert(0!=towers);
   reco::PFCandidateCollection::const_iterator tower = towers->begin();
   std::advance(tower, itemIdx);

   eta = tower->eta();
   phi = tower->phi(); 
   }


