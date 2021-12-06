#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

typedef Merger<reco::TrackCollection> TrackSimpleMerger;

DEFINE_FWK_MODULE(TrackSimpleMerger);
