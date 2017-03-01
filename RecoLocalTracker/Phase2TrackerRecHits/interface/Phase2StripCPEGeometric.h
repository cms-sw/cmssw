#ifndef RecoLocalTracker_Phase2TrackerRecHits_Phase2StripCPEGeometric_H
#define RecoLocalTracker_Phase2TrackerRecHits_Phase2StripCPEGeometric_H

#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


class Phase2StripCPEGeometric : public ClusterParameterEstimator<Phase2TrackerCluster1D> {

  public:

    Phase2StripCPEGeometric() {};
    Phase2StripCPEGeometric(edm::ParameterSet & conf);
    LocalValues localParameters(const Phase2TrackerCluster1D & cluster, const GeomDetUnit & det) const;

};


#endif
