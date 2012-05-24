#include "DataFormats/METReco/interface/MVAMETData.h"

namespace reco {

bool operator<(const JetInfo& jet1, const JetInfo& jet2)
{
  return jet1.p4_.pt() > jet2.p4_.pt();
}

}
