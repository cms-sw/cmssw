#include "DataFormats/MuonReco/interface/Muon.h"
#include <algorithm>

namespace reco {
    typedef std::pair<const reco::Muon *, const reco::Muon *> MuonPointerPair;
}

namespace { struct dictionary  {  // apparenlty better than namespace { namespace {
    reco::MuonPointerPair pair;
}; }
