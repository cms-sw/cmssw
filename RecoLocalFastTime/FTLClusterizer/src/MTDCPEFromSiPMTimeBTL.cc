#include "RecoLocalFastTime/FTLClusterizer/interface/BTLRecHitsErrorEstimatorIM.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDCPEFromSiPMTimeBTL.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

using namespace std;

//-----------------------------------------------------------------------------
//  A constructor run for generic and templates
//
//-----------------------------------------------------------------------------
MTDCPEFromSiPMTimeBTL::MTDCPEFromSiPMTimeBTL(edm::ParameterSet const& conf, const MTDGeometry& geom)
    : MTDCPEBase(conf, geom) {}

LocalPoint MTDCPEFromSiPMTimeBTL::localPosition(DetParam const& dp, ClusterParam& cp) const {
  MeasurementPoint pos(cp.theCluster->x(), cp.theCluster->y());
  return dp.theTopol->localPosition(pos);
}

LocalError MTDCPEFromSiPMTimeBTL::localError(DetParam const& dp, ClusterParam& cp) const {
  constexpr double one_over_twelve = 1. / 12.;
  MeasurementPoint pos(cp.theCluster->x(), cp.theCluster->y());
  MeasurementError simpleRect(one_over_twelve, 0, one_over_twelve);
  if (GeomDetEnumerators::isBarrel(dp.thePart)) {
    LocalPoint lp = localPosition(dp, cp);
    BTLRecHitsErrorEstimatorIM btlHitErr(dp.theDet, lp);
    LocalError err_improved = btlHitErr.localError();
    return err_improved;
  } else {
    return dp.theTopol->localError(pos, simpleRect);
  }
}
