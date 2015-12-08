#include "DataFormats/TrackReco/interface/Track.h"
namespace PFTrackAlgoTools  {
//used in general tracks importer and in PFAlgo
  unsigned int getAlgoCategory(const reco::TrackBase::TrackAlgorithm& algo,bool hltIterativeTracking  = true );
}
