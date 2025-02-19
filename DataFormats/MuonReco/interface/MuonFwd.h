#ifndef MuonReco_MuonFwd_h
#define MuonReco_MuonFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class Muon;
  /// collection of Muon objects
  typedef std::vector<Muon> MuonCollection;
  /// presistent reference to a Muon
  typedef edm::Ref<MuonCollection> MuonRef;
  /// references to Muon collection
  typedef edm::RefProd<MuonCollection> MuonRefProd;
  /// vector of references to Muon objects all in the same collection
  typedef edm::RefVector<MuonCollection> MuonRefVector;
  /// iterator over a vector of references to Muon objects all in the same collection
  typedef MuonRefVector::iterator muon_iterator;

  /// Links between the three tracks which can define a muon
  class MuonTrackLinks;
  
  /// collection of MuonTrackLinks
  typedef std::vector<MuonTrackLinks> MuonTrackLinksCollection;
  
  class CaloMuon;
  /// collection of Muon objects
  typedef std::vector<CaloMuon> CaloMuonCollection;
}

#endif
