#ifndef MuonReco_MuonFwd_h
#define MuonReco_MuonFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class Muon;
  typedef std::vector<Muon> MuonCollection;
  typedef edm::Ref<MuonCollection> MuonRef;
  typedef edm::RefVector<MuonCollection> MuonRefs;
  typedef MuonRefs::iterator muon_iterator;
}

#endif
