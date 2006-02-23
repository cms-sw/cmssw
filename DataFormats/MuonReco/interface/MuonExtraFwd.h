#ifndef MuonReco_MuonExtraFwd_h
#define MuonReco_MuonExtraFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class MuonExtra;
  typedef std::vector<MuonExtra> MuonExtraCollection;
  typedef edm::Ref<MuonExtraCollection> MuonExtraRef;
  typedef edm::RefVector<MuonExtraCollection> MuonExtraRefs;
  typedef MuonExtraRefs::iterator muonExtra_iterator;
}

#endif
