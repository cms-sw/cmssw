#include "Fireworks/Candidates/interface/FWCandidateTowerSliceSelector.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"


FWCandidateTowerSliceSelector::FWCandidateTowerSliceSelector(TH2F* h, const FWEventItem* i):
   FWHistSliceSelector(h, i)
{
}


FWCandidateTowerSliceSelector::~FWCandidateTowerSliceSelector()
{
}
void
FWCandidateTowerSliceSelector::getItemEntryEtaPhi(int itemIdx, float& eta, float& phi) const
{
   
   const reco::CandidateCollection* towers=0;
   m_item->get(towers);
   assert(0!=towers);
   reco::CandidateCollection::const_iterator tower = towers->begin();
   std::advance(tower, itemIdx);

   eta = tower->eta();
   phi = tower->phi(); 
}


