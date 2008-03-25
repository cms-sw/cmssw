#ifndef MuonIsolation_TrackSelector_H
#define MuonIsolation_TrackSelector_H

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/MuonReco/interface/Direction.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/View.h"
#include <list>

namespace muonisolation {

class TrackSelector {
public:

  typedef muonisolation::Range<float> Range;
  typedef std::list<const reco::Track*> result_type;
  typedef edm::View<reco::Track> input_type;

  TrackSelector(const Range & z, float r, const Direction &dir, float drMax);
  result_type operator()(const input_type & tracks) const;


private:
  Range theZ;
  Range theR;
  Direction theDir;
  float theDR_Max;

}; 

}

#endif 
