#ifndef BTauReco_TraclCountingTagInfoFwd_h
#define BTauReco_TraclCountingTagInfoFwd_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class SoftLeptonTagInfo;
  typedef std::vector<SoftLeptonTagInfo>                SoftLeptonTagInfoCollection;
  typedef edm::Ref<SoftLeptonTagInfoCollection>         SoftLeptonTagInfoRef;
  typedef edm::RefProd<SoftLeptonTagInfoCollection>     SoftLeptonTagInfoRefProd;
  typedef edm::RefVector<SoftLeptonTagInfoCollection>   SoftLeptonTagInfoRefVector;
}

#endif // BTauReco_TraclCountingTagInfoFwd_h
