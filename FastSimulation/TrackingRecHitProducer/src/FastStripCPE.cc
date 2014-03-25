#include "FastStripCPE.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <algorithm>
#include<cmath>
#include<map>
#include <memory>

void FastStripCPE::enterLocalParameters(uint32_t id, uint16_t firstStrip, const std::pair<LocalPoint, LocalError>& pos_err_info ) {
  //Filling the map.
  pos_err_map.insert(std::make_pair(std::make_pair(id, firstStrip), pos_err_info));
}


StripClusterParameterEstimator::LocalValues FastStripCPE::localParameters( const SiStripCluster & cl,const GeomDetUnit& det) const {
  // strip cluster never supported detID
  std::map<std::pair<uint32_t,uint16_t>,std::pair<LocalPoint,LocalError> >::const_iterator strip_link = pos_err_map.find(std::make_pair(det.geographicalId(),cl.firstStrip()));
  if (strip_link != pos_err_map.end()) {
    std::pair<LocalPoint,LocalError> pos_err_info = strip_link->second;
    LocalPoint result = pos_err_info.first;
    LocalError eresult = pos_err_info.second;
    return std::make_pair(result,eresult);
  }
  throw cms::Exception("FastStripCPE") << "Cluster not filled.";
}

std::unique_ptr<ClusterParameterEstimator<SiStripCluster>>
FastStripCPE::clone() const {
    return std::unique_ptr<ClusterParameterEstimator<SiStripCluster>>(new ClusterParameterEstimator<SiStripCluster>(*this));
}

LocalVector FastStripCPE::driftDirection(const StripGeomDetUnit* det) const {
  throw cms::Exception("FastStripCPE") << "Should Not Be Called.";
}
