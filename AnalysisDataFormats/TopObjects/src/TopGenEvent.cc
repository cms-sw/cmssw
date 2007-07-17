#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"

const reco::Candidate*
TopGenEvent::candidate(int id) const
{
  const reco::Candidate* cand=0;
  const reco::CandidateCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (partsColl[i].pdgId()==id) {
      cand = &partsColl[i];
    }
  }  
  return cand;
}
