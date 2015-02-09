#include "Fireworks/Candidates/interface/FWCandidateTowerSliceSelector.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWSimpleProxyHelper.h"


FWCandidateTowerSliceSelector::FWCandidateTowerSliceSelector(TH2F* h, const FWEventItem* i, FWSimpleProxyHelper* helper):
   FWHistSliceSelector(h, i),
   m_helper(helper)
{
}


FWCandidateTowerSliceSelector::~FWCandidateTowerSliceSelector()
{
}
void
FWCandidateTowerSliceSelector::getItemEntryEtaPhi(int itemIdx, float& eta, float& phi) const
{
   const void* modelData = m_item->modelData(itemIdx);
   if (modelData) {
      const reco::Candidate* tower = reinterpret_cast<const reco::Candidate*>(m_helper->offsetObject(modelData));
      eta = tower->eta();
      phi = tower->phi();
   }
}


