#include "DataFormats/METReco/interface/PUSubMETData.h"

namespace reco 
{
  bool operator<(const reco::PUSubMETCandInfo& jet1, const reco::PUSubMETCandInfo& jet2)
  {
    return jet1.p4_.pt() > jet2.p4_.pt();
  }
}
