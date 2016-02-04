#ifndef RecoLocalTracker_Phase2TrackerRecHits_Phase2StripCPETrivial_H
#define RecoLocalTracker_Phase2TrackerRecHits_Phase2StripCPETrivial_H

#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


class Phase2StripCPETrivial : public ClusterParameterEstimator<Phase2TrackerCluster1D> {

  public:

    LocalValues localParameters(const Phase2TrackerCluster1D & cluster, const GeomDetUnit & det) const;

};


#endif
