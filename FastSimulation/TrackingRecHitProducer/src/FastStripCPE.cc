#include "FastStripCPE.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <algorithm>
#include<cmath>
#include <memory>


StripClusterParameterEstimator::LocalValues FastStripCPE::localParameters( const SiStripCluster & cl,const GeomDetUnit& det) const {
   //the information should be retrieved from the cluster itself.
   return StripClusterParameterEstimator::LocalValues();
}

LocalVector FastStripCPE::driftDirection(const StripGeomDetUnit* det) const {
  throw cms::Exception("FastStripCPE") << "Should Not Be Called.";
}
