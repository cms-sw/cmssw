#ifndef EgammaReco_SiStripElectronCandidateFwd_h
#define EgammaReco_SiStripElectronCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class SiStripElectronCandidate;

  /// collectin of SiStripElectronCandidate objects
  typedef std::vector<SiStripElectronCandidate> SiStripElectronCandidateCollection;

  /// reference to an object in a collection of SiStripElectronCandidate objects
  typedef edm::Ref<SiStripElectronCandidateCollection> SiStripElectronCandidateRef;

  /// reference to a collection of SiStripElectronCandidate objects
  typedef edm::RefProd<SiStripElectronCandidateCollection> SiStripElectronCandidateRefProd;

  /// vector of objects in the same collection of SiStripElectronCandidate objects
  typedef edm::RefVector<SiStripElectronCandidateCollection> SiStripElectronCandidateRefVector;

  /// iterator over a vector of reference to SiStripElectronCandidate objects
  typedef SiStripElectronCandidateRefVector::iterator siStripElectronCandidate_iterator;
}

#endif
