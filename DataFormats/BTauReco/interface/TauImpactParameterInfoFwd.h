#ifndef BTauReco_TauImpactParameterInfoFwd_h
#define BTauReco_TauImpactParameterInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class TauImpactParameterTrackData;
  class TauImpactParameterInfo;
  typedef std::vector<TauImpactParameterInfo> TauImpactParameterInfoCollection;
  typedef edm::Ref<TauImpactParameterInfoCollection> TauImpactParameterInfoRef;
  typedef edm::RefProd<TauImpactParameterInfoCollection> TauImpactParameterInfoRefProd;
  typedef edm::RefVector<TauImpactParameterInfoCollection> TauImpactParameterInfoRefVector;
}

#endif // BTauReco_TauImpactParameterInfoFwd_h
