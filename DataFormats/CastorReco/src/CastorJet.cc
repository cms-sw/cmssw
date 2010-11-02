#include "DataFormats/CastorReco/interface/CastorJet.h"

using namespace reco;
using namespace edm;

CastorJet::CastorJet(const double energycal, const CastorClusterRef& usedCluster) {
  energycal_ = energycal;
  usedCluster_ = usedCluster;
}

CastorJet::~CastorJet() {

}
