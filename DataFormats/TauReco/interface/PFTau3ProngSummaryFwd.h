#ifndef DataFormats_TauReco_PFTau3ProngSummaryFwd_h
#define DataFormats_TauReco_PFTau3ProngSummaryFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class PFTau3ProngSummary;
  /// collection of PFTau3ProngSummary objects
  typedef std::vector<reco::PFTau3ProngSummary> PFTau3ProngSummaryCollection;
  /// presistent reference to a PFTau3ProngSummary
  typedef edm::Ref<PFTau3ProngSummaryCollection> PFTau3ProngSummaryRef;
  /// references to PFTau3ProngSummary collection
  typedef edm::RefProd<PFTau3ProngSummaryCollection> PFTau3ProngSummaryRefProd;
  /// vector of references to PFTau3ProngSummary objects all in the same collection
  typedef edm::RefVector<PFTau3ProngSummaryCollection> PFTau3ProngSummaryRefVector;
  /// iterator over a vector of references to PFTau3ProngSummary objects all in the same collection
  typedef PFTau3ProngSummaryRefVector::iterator PFTau3ProngSummary_iterator;
}

#endif
