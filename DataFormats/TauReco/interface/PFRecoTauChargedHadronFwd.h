#ifndef DataFormats_TauReco_PFRecoTauChargedHadronFwd_h
#define DataFormats_TauReco_PFRecoTauChargedHadronFwd_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ptr.h"

#include <vector>

namespace reco {
  class PFRecoTauChargedHadron;
  /// collection of PFRecoTauChargedHadron objects
  typedef std::vector<PFRecoTauChargedHadron> PFRecoTauChargedHadronCollection;
  /// presistent reference to a PFRecoTauChargedHadron
  typedef edm::Ref<PFRecoTauChargedHadronCollection> PFRecoTauChargedHadronRef;
  /// references to PFRecoTauChargedHadron collection
  typedef edm::RefProd<PFRecoTauChargedHadronCollection> PFRecoTauChargedHadronRefProd;
  /// vector of references to PFRecoTauChargedHadron objects all in the same collection
  typedef edm::RefVector<PFRecoTauChargedHadronCollection> PFRecoTauChargedHadronRefVector;
  /// iterator over a vector of references to PFRecoTauChargedHadron objects all in the same collection
  typedef PFRecoTauChargedHadronRefVector::iterator PFRecoTauChargedHadronRefVector_iterator;
}

#endif
