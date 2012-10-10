#include "FastPixelCPE.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"


// Services
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "boost/multi_array.hpp"

#include <iostream>
#include <map>
using namespace std;

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------


PixelClusterParameterEstimator::LocalValues FastPixelCPE::localParameters(const SiPixelCluster & cluster, const GeomDetUnit &det) const
{
  
  std::map<std::pair<unsigned int, std::pair<int,int> >, std::pair<LocalPoint,LocalError> >::const_iterator pixel_link = 
    pos_err_map.find(std::make_pair(det.geographicalId().rawId(),std::make_pair(cluster.minPixelRow(),cluster.minPixelCol())));
  if (pixel_link != pos_err_map.end()) {
    std::pair<LocalPoint,LocalError> pos_err_info = pixel_link->second;
    return std::make_pair(pos_err_info.first, pos_err_info.second);
  }
  throw cms::Exception("FastPixelCPE") << "Cluster not filled.";
}

void FastPixelCPE::enterLocalParameters(unsigned int id, std::pair<int,int> &row_col, std::pair<LocalPoint,LocalError> pos_err_info) const {
  pos_err_map.insert(std::make_pair(std::make_pair(id,row_col), pos_err_info));
}
