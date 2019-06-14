#ifndef ParticleFlowReco_PFConversionFwd_h
#define ParticleFlowReco_PFConversionFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class PFConversion;

  /// collection of PFConversion objects
  typedef std::vector<PFConversion> PFConversionCollection;

  /// persistent reference to PFConversion objects
  typedef edm::Ref<PFConversionCollection> PFConversionRef;

  /// reference to PFConversion collection
  typedef edm::RefProd<PFConversionCollection> PFConversionRefProd;

  /// vector of references to PFConversion objects all in the same collection
  typedef edm::RefVector<PFConversionCollection> PFConversionRefVector;

  /// iterator over a vector of references to PFConversion objects
  typedef PFConversionRefVector::iterator PFConversion_iterator;
}  // namespace reco

#endif
