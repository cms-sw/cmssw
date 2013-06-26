#include "DataFormats/METReco/interface/MVAMEtData.h"

namespace reco 
{
  bool operator<(const MVAMEtJetInfo& jet1, const MVAMEtJetInfo& jet2)
  {
    return jet1.p4_.pt() > jet2.p4_.pt();
  }
}
