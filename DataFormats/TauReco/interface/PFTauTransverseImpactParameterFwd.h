#ifndef DataFormats_TauReco_PFTauTransverseImpactParameterFwd_h
#define DataFormats_TauReco_PFTauTransverseImpactParameterFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class PFTauTransverseImpactParameter;
  /// collection of PFTauTransverseImpactParameter objects
  typedef std::vector<reco::PFTauTransverseImpactParameter> PFTauTransverseImpactParameterCollection;
  /// presistent reference to a PFTauTransverseImpactParameter
  typedef edm::Ref<PFTauTransverseImpactParameterCollection> PFTauTransverseImpactParameterRef;
  /// references to PFTauTransverseImpactParameter collection
  typedef edm::RefProd<PFTauTransverseImpactParameterCollection> PFTauTransverseImpactParameterRefProd;
  /// vector of references to PFTauTransverseImpactParameter objects all in the same collection
  typedef edm::RefVector<PFTauTransverseImpactParameterCollection> PFTauTransverseImpactParameterRefVector;
  /// iterator over a vector of references to PFTauTransverseImpactParameter objects all in the same collection
  typedef PFTauTransverseImpactParameterRefVector::iterator PFTauTransverseImpactParameter_iterator;
}

#endif
