#ifndef RecoLocalTracker_Phase2TrackerRecHits_Phase2StripCPE_H
#define RecoLocalTracker_Phase2TrackerRecHits_Phase2StripCPE_H

#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class Phase2StripCPE final : public ClusterParameterEstimator<Phase2TrackerCluster1D> {
public:
  // currently (?) use Pixel classes for GeomDetUnit and Topology
  using Phase2TrackerGeomDetUnit = PixelGeomDetUnit;
  using Phase2TrackerTopology = PixelTopology;

  struct Param {
    Param() : topology(nullptr) {}
    Phase2TrackerTopology const* topology;
    LocalError localErr;
    float coveredStrips;
  };

public:
  Phase2StripCPE(edm::ParameterSet& conf, const MagneticField&, const TrackerGeometry&);
  LocalValues localParameters(const Phase2TrackerCluster1D& cluster, const GeomDetUnit& det) const override;
  LocalVector driftDirection(const Phase2TrackerGeomDetUnit& det) const;

private:
  void fillParam();
  std::vector<Param> m_Params;

  const MagneticField& magfield_;
  const TrackerGeometry& geom_;
  float tanLorentzAnglePerTesla_;
  unsigned int m_off;

  bool use_LorentzAngle_DB_;
};

#endif
