#ifndef EgammaReco_ConversionFwd_h
#define EgammaReco_ConversionFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class Conversion;

  /// collectin of Conversion objects
  typedef std::vector<Conversion> ConversionCollection;

  /// reference to an object in a collection of Conversion objects
  typedef edm::Ref<ConversionCollection> ConversionRef;

  /// reference to a collection of Conversion objects
  typedef edm::RefProd<ConversionCollection> ConversionRefProd;

  /// vector of objects in the same collection of Conversion objects
  typedef edm::RefVector<ConversionCollection> ConversionRefVector;

  /// iterator over a vector of reference to Conversion objects
  typedef ConversionRefVector::iterator c_iterator;
}

#endif
