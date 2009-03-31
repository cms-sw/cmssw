#include "RecoMuon/TrackerSeedGenerator/interface/L1MuonSeedsMerger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"

L1MuonSeedsMerger::L1MuonSeedsMerger(const edm::ParameterSet& cfg)
{
  theDeltaEtaCut  = cfg.getParameter<double>("deltaEtaCut");
  theDiffRelPtCut = cfg.getParameter<double>("diffRelPtCut");
}

void L1MuonSeedsMerger::resolve(std::vector<TrackAndHits>& tracks) const
{
  sort(tracks.begin(),tracks.end(), Less());
  typedef std::vector<TrackAndHits>::iterator Tracks_Itr; 
  Tracks_Itr it1 = tracks.begin();
  while (it1 != tracks.end() ) {
    for (Tracks_Itr it2 = it1+1; it1->first && it2<tracks.end(); it2++) {
      if (! it2->first) continue;
      if ( it2->first->eta() - it1->first->eta() > theDeltaEtaCut) break;
      switch ( compare( &(*it1), &(*it2) ) ) {
         case killFirst :
           delete it1->first;
           it1->first = 0;
           break;
         case killSecond :
           delete it2->first;
           it2->first = 0;
           break;
         case mergeTwo :
           *it2 = *(merge(&(*it1),&(*it2)));
           it1->first = 0;
           break;
         case goAhead : default: break;
      }
    }
    if (0 == it1->first) tracks.erase(it1); else it1++;
  }
}

bool L1MuonSeedsMerger::Less::operator()(const TrackAndHits& a, const TrackAndHits& b) const
{
  return (a.first->eta() < b.first->eta());
}

const L1MuonSeedsMerger::TrackAndHits* 
    L1MuonSeedsMerger::merge(const TrackAndHits* a, const TrackAndHits* b) const
{
// temporary algorith, takes track with bigger pt, to be reimplemented
  if (a->first->pt() > b->first->pt()) {
    delete b->first;
    return a;
  } else {
    delete a->first;
    return b;
  }
}

L1MuonSeedsMerger::Action
    L1MuonSeedsMerger::compare(const TrackAndHits* a, const TrackAndHits* b) const
{
  int nshared = 0;
  const SeedingHitSet & hitsA = a->second;
  const SeedingHitSet & hitsB = b->second;
  for (unsigned int iHitA=0, nHitsA=hitsA.size(); iHitA < nHitsA; ++iHitA) {
    const TrackingRecHit* trhA = hitsA[iHitA]->hit();
    for (unsigned int iHitB=0, nHitsB=hitsB.size(); iHitB < nHitsB; ++iHitB) {
      const TrackingRecHit* trhB = hitsB[iHitB]->hit();
      if (trhA==trhB) nshared++;
    }
  }

  if (nshared >= 2) {
    if (hitsA.size() >= 3 && hitsB.size() >= 3 )
      return (a->first->chi2() > b->first->chi2()) ? killFirst : killSecond;
    else if (hitsB.size() >= 3)
      return killFirst;
    else
      return killSecond;
  }
  else if (nshared >= 1) {
    if (hitsA.size() != hitsB.size())
      return (hitsA.size() < hitsB.size()) ? killFirst : killSecond;
    else if (    hitsA.size() >= 3
              && a->first->charge()==b->first->charge()
              && fabs(a->first->pt()-b->first->pt())/b->first->pt() < theDiffRelPtCut )
      return (a->first->chi2() > b->first->chi2()) ? killFirst : killSecond;
    else if (    hitsA.size() == 2 )
      return (a->first->pt() < b->first->pt()) ? killFirst : killSecond;
    else
      return goAhead;
  }
  else return goAhead;
}

