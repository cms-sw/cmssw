#include "DataFormats/CastorReco/interface/CastorEgamma.h"

reco::CastorEgamma::CastorEgamma(const double energycal, const reco::CastorClusterRef& usedCluster) {
  energycal_ = energycal;
  usedCluster_ = usedCluster;
}

reco::CastorEgamma::~CastorEgamma() {

}
