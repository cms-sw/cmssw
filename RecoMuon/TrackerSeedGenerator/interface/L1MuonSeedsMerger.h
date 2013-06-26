#ifndef RecoMuon_TrackerSeedGenerator_L1MuonSeedsMerger_H
#define RecoMuon_TrackerSeedGenerator_L1MuonSeedsMerger_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <vector>

namespace edm {class ParameterSet;}

class L1MuonSeedsMerger {
public:
  typedef std::pair<const reco::Track*, SeedingHitSet > TrackAndHits; 
  typedef std::vector<TrackAndHits> TracksAndHits; 
  L1MuonSeedsMerger(const edm::ParameterSet& cfg);
  virtual ~L1MuonSeedsMerger(){}
  virtual void resolve(TracksAndHits &) const;
private:
  enum Action { goAhead, killFirst, killSecond, mergeTwo };
  struct Less { bool operator()(const TrackAndHits&, const TrackAndHits&) const; };
  const TrackAndHits*  merge(const TrackAndHits*,const TrackAndHits*) const;
  Action compare(const TrackAndHits*, const TrackAndHits*) const;
private:
  float theDeltaEtaCut;
  float theDiffRelPtCut;
};
#endif
