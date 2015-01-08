#include "Fireworks/Candidates/interface/FWCandidateTowerSliceSelector.h"
//#include "DataFormats/Candidate/interface/CandidateFwd.h"
//#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

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
   
   const pat::PackedCandidateCollection* towers=0;
   m_item->get(towers);
   assert(0!=towers);
   pat::PackedCandidateCollection::const_iterator tower = towers->begin();
   std::advance(tower, itemIdx);

   eta = tower->eta();
   phi = tower->phi(); 
}


