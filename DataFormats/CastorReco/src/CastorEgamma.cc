#include "DataFormats/CastorReco/interface/CastorEgamma.h"

using namespace reco;
using namespace edm;

CastorEgamma::CastorEgamma(const double energycal, const CastorClusterRef& usedCluster) {
  energycal_ = energycal;
  usedCluster_ = usedCluster;
}

CastorEgamma::~CastorEgamma() {

}
