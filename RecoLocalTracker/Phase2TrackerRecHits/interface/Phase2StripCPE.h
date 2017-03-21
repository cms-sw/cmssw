#ifndef RecoLocalTracker_Phase2TrackerRecHits_Phase2StripCPE_H
#define RecoLocalTracker_Phase2TrackerRecHits_Phase2StripCPE_H

#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


class Phase2StripCPE : public ClusterParameterEstimator<Phase2TrackerCluster1D> {

  public:

    // currently (?) use Pixel classes for GeomDetUnit and Topology
    using Phase2TrackerGeomDetUnit = PixelGeomDetUnit;
    using Phase2TrackerTopology = PixelTopology ;

  public:

    Phase2StripCPE() {};
    Phase2StripCPE(edm::ParameterSet & conf, const MagneticField &);
    LocalValues localParameters(const Phase2TrackerCluster1D & cluster, const GeomDetUnit & det) const;
    LocalVector driftDirection(const Phase2TrackerGeomDetUnit & det) const;

  protected:

    const MagneticField * magfield_;
    bool use_LorentzAngle_DB_;
    double tanLorentzAnglePerTesla_;

};


#endif
