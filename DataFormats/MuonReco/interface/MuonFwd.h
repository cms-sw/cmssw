#ifndef MuonReco_MuonFwd_h
#define MuonReco_MuonFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  namespace io_v1 {
    class Muon;
  }
  using Muon = io_v1::Muon;
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
  namespace io_v1 {
    class MuonTrackLinks;
  }
  using MuonTrackLinks = io_v1::MuonTrackLinks;
  /// collection of MuonTrackLinks
  typedef std::vector<MuonTrackLinks> MuonTrackLinksCollection;
  namespace io_v1 {
    class CaloMuon;
  }
  using CaloMuon = io_v1::CaloMuon;
  /// collection of Muon objects
  typedef std::vector<CaloMuon> CaloMuonCollection;
}  // namespace reco

#endif
