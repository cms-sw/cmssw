#ifndef BTauReco_TauMassTagInfoFwd_h
#define BTauReco_TauMassTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class TauImpactParameterTrackData;
  class TauMassTagInfo;
  typedef std::vector<TauMassTagInfo> TauMassTagInfoCollection;
  typedef edm::Ref<TauMassTagInfoCollection> TauMassTagInfoRef;
  typedef edm::RefProd<TauMassTagInfoCollection> TauMassTagInfoRefProd;
  typedef edm::RefVector<TauMassTagInfoCollection> TauMassTagInfoRefVector;
}

#endif // BTauReco_TauMassTagInfoFwd_h
