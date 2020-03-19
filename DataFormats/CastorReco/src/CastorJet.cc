#include "DataFormats/CastorReco/interface/CastorJet.h"

reco::CastorJet::CastorJet(const double energycal, const reco::CastorClusterRef& usedCluster) {
  energycal_ = energycal;
  usedCluster_ = usedCluster;
}

reco::CastorJet::~CastorJet() {}
