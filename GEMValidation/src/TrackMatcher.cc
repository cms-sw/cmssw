#include "GEMCode/GEMValidation/src/TrackMatcher.h"

TrackMatcher::TrackMatcher() 
{
}

TrackMatcher::TrackMatcher(const TrackMatcher& rhs) 
{
}

TrackMatcher::~TrackMatcher() 
{
}


TFTrack* 
TrackMatcher::bestTFTrack(bool sortPtFirst)
{
  return 0;
  /*
  if (tfTracks_.size()==0) return NULL;
  
  // determine max # of matched stubs in the TFTrack collection
  int maxNMatchedMPC = 0;
  for (unsigned i=0; i<tfTracks_.size(); i++) {
    int nst=0;
    for (size_t s=0; s<tfTracks_.at(i).ids.size(); s++) 
      if (tfTracks_.at(i).mplcts[s]->deltaOk) nst++;
    if (nst>maxNMatchedMPC) maxNMatchedMPC = nst;
  }
  // collect tracks with max # of matched stubs
  std::vector<unsigned> bestMatchedTracks;
  for (unsigned i=0; i<tfTracks_.size(); i++) {
    int nst=0;
    for (size_t s=0; s<tfTracks_.at(i).ids.size(); s++) 
      if (tfTracks_.at(i).mplcts[s]->deltaOk) nst++;
    if (nst==maxNMatchedMPC) bestMatchedTracks.push_back(i);
  }
  
  // already found the best TFTrack
  if (bestMatchedTracks.size()==1) return &(tfTracks_[bestMatchedTracks[0]]);
  
  // case when you have more than 1 best TFTrack
  // first sort by quality
  double qBase  = 1000000.;
  // then sort by Pt inside the cone (if sortPtFirst), then sort by DR
  double ptBase = 0.;
  if (sortPtFirst) ptBase = 1000.;
  unsigned maxI = 99;
  double maxRank = -999999.;
  for (unsigned i=0; i<tfTracks_.size(); i++) {
    if (bestMatchedTracks.size()) {
      bool gotit=0;
      for (unsigned m=0;m<bestMatchedTracks.size();m++) if (bestMatchedTracks[m]==i) gotit=1;
      if (!gotit) continue;
    }
    double rr = qBase*tfTracks_.at(i).q_packed + ptBase*tfTracks_.at(i).pt_packed + 1./(0.01 + tfTracks_.at(i).dr);
    if (rr > maxRank) { maxRank = rr; maxI = i;}
  }
  if (maxI==99) return NULL;
  return &(tfTracks_.at(maxI));
  */
}


TFCand* 
TrackMatcher::bestTFCand(bool sortPtFirst)
{
  /*
    if (cands.size()==0) return NULL;

    // determine max # of matched stubs
    int maxNMatchedMPC = 0;
    for (unsigned i=0; i<cands.size(); i++) 
    {
        int nst=0;
        if (cands[i].tftrack==0) continue;
        for (size_t s=0; s<cands[i].tftrack->ids.size(); s++) 
            if (cands[i].tftrack->mplcts[s]->deltaOk) nst++;
        if (nst>maxNMatchedMPC) maxNMatchedMPC = nst;
    }

    // collect tracks with max # of matched stubs
    std::vector<unsigned> bestMatchedTracks;
    if (maxNMatchedMPC>0) {
        for (unsigned i=0; i<cands.size(); i++) 
        {
            int nst=0;
            if (cands[i].tftrack==0) continue;
            for (size_t s=0; s<cands[i].tftrack->ids.size(); s++) 
                if (cands[i].tftrack->mplcts[s]->deltaOk) nst++;
            if (nst==maxNMatchedMPC) bestMatchedTracks.push_back(i);
        }
        if (bestMatchedTracks.size()==1) return &(cands[bestMatchedTracks[0]]);
    }

    // first sort by quality
    double qBase  = 1000000.;
    // then sort by Pt inside the cone (if sortPtFirst), then sort by DR
    double ptBase = 0.;
    if (sortPtFirst) ptBase = 1000.;
    unsigned maxI = 99;
    double maxRank = -999999.;
    for (unsigned i=0; i<cands.size(); i++) 
    {
        if (bestMatchedTracks.size()) {
            bool gotit=0;
            for (unsigned m=0;m<bestMatchedTracks.size();m++) if (bestMatchedTracks[m]==i) gotit=1;
            if (!gotit) continue;
        }
        // quality criterium you apply to get the best TFCand
        double rr = qBase*cands[i].l1cand->quality_packed() + ptBase*cands[i].l1cand->pt_packed() + 1./(0.01 + cands[i].dr);
        if (rr > maxRank) { maxRank = rr; maxI = i;}
    }
    if (maxI==99) return NULL;
    return &(cands[maxI]);
  */
  return 0;
}

GMTRegCand* 
TrackMatcher::bestGMTRegCand(bool sortPtFirst)
{
  // first sort by Pt inside the cone (if sortPtFirst), then sort by DR
  if (gmtRegCands_.size()==0) return nullptr;
  const double ptBase(sortPtFirst ? 1000. : 0.);
  unsigned maxI = 99;
  double maxRank = -999999.;
  for (unsigned i=0; i<gmtRegCands_.size(); i++) {
    // quality criterium to sort the GMT Regional candidates
    const double rank(ptBase*gmtRegCands_.at(i)->pt() + 1./(0.01 + gmtRegCands_.at(i)->dr()));
    if (rank > maxRank) { 
      maxRank = rank; 
      maxI = i;
    }
  }
  if (maxI==99) return nullptr;
  return gmtRegCands_.at(maxI);
}

GMTCand* 
TrackMatcher::bestGMTCand(bool sortPtFirst)
{
  // first sort by Pt inside the cone (if sortPtFirst), then sort by DR
  if (gmtCands_.size()==0) return nullptr;
  const double ptBase(sortPtFirst ? 1000. : 0.);
  unsigned maxI = 99;
  double maxRank = -999999.;
  for (unsigned i=0; i<gmtCands_.size(); i++) {
    // quality criterium to sort the GMT candidates
    const double rank(ptBase*gmtCands_.at(i)->pt() + 1./(0.01 + gmtCands_.at(i)->dr()));
    if (rank > maxRank) { 
      maxRank = rank; 
      maxI = i;
    }
  }
  if (maxI==99) return nullptr;
  return gmtCands_.at(maxI);
}

L1Extra* 
TrackMatcher::bestL1Extra(bool sortPtFirst)
{
  return 0;
}
