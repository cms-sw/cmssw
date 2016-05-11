
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPETrivial.h"


Phase2StripCPETrivial::LocalValues Phase2StripCPETrivial::localParameters(const Phase2TrackerCluster1D & cluster, const GeomDetUnit & det) const {
  float strippitch  = 0.0090; // hardcoded dummy, a la 2S
  float striplength = 5.;     // hardcoded dummy, a la 2S
  std::pair<float, float> barycenter = cluster.barycenter();
  LocalPoint lp( barycenter.second * strippitch, striplength * (barycenter.first + 1/2), 0 );
  LocalError le( pow(strippitch,2) / 12, pow(striplength,2) / 12, 0 );
  return std::make_pair( lp, le );
}


// needed, otherwise linker misses some refs
#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(ClusterParameterEstimator<Phase2TrackerCluster1D>);
