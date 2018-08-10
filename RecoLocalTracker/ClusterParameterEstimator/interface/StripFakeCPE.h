#ifndef RecoLocalTracker_Fake_StripCluster_Parameter_Estimator_H
#define RecoLocalTracker_Fake_StripCluster_Parameter_Estimator_H

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"

#include "CommonTools/Utils/interface/DynArray.h"
#include "FWCore/Utilities/interface/Exception.h"


#include "RecoLocalTracker/ClusterParameterEstimator/interface/FakeCPE.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"

class StripFakeCPE final : public StripClusterParameterEstimator
{
 public:

  StripFakeCPE() = default; 
  ~StripFakeCPE() override = default;

  using LocalValues = std::pair<LocalPoint,LocalError>;

  LocalValues localParameters( const SiStripCluster& cl , const GeomDetUnit& gd) const override {
     return fakeCPE().map().get(cl,gd);
  }

  // used by Validation....
  LocalVector driftDirection(const StripGeomDetUnit* ) const override { return LocalVector();}

  void setFakeCPE(FakeCPE * iFakeCPE) { m_fakeCPE = iFakeCPE;}
  FakeCPE const & fakeCPE() const { return *m_fakeCPE; }


private:

  FakeCPE const * m_fakeCPE=nullptr;

  
};


#endif




