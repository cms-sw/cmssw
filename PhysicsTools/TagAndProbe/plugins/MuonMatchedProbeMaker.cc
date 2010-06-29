
#include "DataFormats/MuonReco/interface/Muon.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/TagAndProbe/interface/MatchedProbeMaker.h"

typedef MatchedProbeMaker< reco::Muon > MuonMatchedProbeMaker;

DEFINE_FWK_MODULE( MuonMatchedProbeMaker );
