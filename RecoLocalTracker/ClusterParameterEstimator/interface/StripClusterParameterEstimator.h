#ifndef RecoLocalTracker_StripCluster_Parameter_Estimator_H
#define RecoLocalTracker_StripCluster_Parameter_Estimator_H

#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"

typedef ClusterParameterEstimator<SiStripCluster> StripClusterParameterEstimator;

#endif
