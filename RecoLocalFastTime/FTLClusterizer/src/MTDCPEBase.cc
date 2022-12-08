#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetUnit.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"

#include "RecoLocalFastTime/FTLClusterizer/interface/MTDCPEBase.h"

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"

#include <iostream>

using namespace std;

//-----------------------------------------------------------------------------
//  A constructor run for generic and templates
//
//-----------------------------------------------------------------------------
MTDCPEBase::MTDCPEBase(edm::ParameterSet const& conf, const MTDGeometry& geom) : geom_(geom) {
  //-----------------------------------------------------------------------------
  //  Fill all variables which are constant for an event (geometry)
  //-----------------------------------------------------------------------------
  auto const& dus = geom_.detUnits();
  unsigned detectors = dus.size();
  m_DetParams.resize(detectors);
  LogDebug("MTDCPEBase::fillDetParams():") << "caching " << detectors << "MTD detectors" << endl;
  for (unsigned i = 0; i != detectors; ++i) {
    auto& p = m_DetParams[i];
    p.theDet = dynamic_cast<const MTDGeomDetUnit*>(dus[i]);
    assert(p.theDet);

    p.theOrigin = p.theDet->surface().toLocal(GlobalPoint(0, 0, 0));

    //--- p.theDet->type() returns a GeomDetType, which implements subDetector()
    p.thePart = p.theDet->type().subDetector();

    //--- bounds() is implemented in BoundSurface itself.
    p.theThickness = p.theDet->surface().bounds().thickness();

    // Cache the det id for templates and generic erros
    p.theTopol = &(static_cast<const ProxyMTDTopology&>(p.theDet->topology()));
    assert(p.theTopol);
    p.theRecTopol = &(static_cast<const RectangularMTDTopology&>(p.theTopol->specificTopology()));
    assert(p.theRecTopol);

    //--- The geometrical description of one module/plaquette
    std::pair<float, float> pitchxy = p.theRecTopol->pitch();
    p.thePitchX = pitchxy.first;   // pitch along x
    p.thePitchY = pitchxy.second;  // pitch along y

    LogDebug("MTDCPEBase::fillDetParams()") << "***** MTD LAYOUT *****"
                                            << " thePart = " << p.thePart << " theThickness = " << p.theThickness
                                            << " thePitchX  = " << p.thePitchX << " thePitchY  = " << p.thePitchY;
  }
}

//------------------------------------------------------------------------
MTDCPEBase::DetParam const& MTDCPEBase::detParam(const GeomDetUnit& det) const { return m_DetParams.at(det.index()); }

LocalPoint MTDCPEBase::localPosition(DetParam const& dp, ClusterParam& cp) const {
  //remember measurement point is row(col)+0.5f
  MeasurementPoint pos(cp.theCluster->x(), cp.theCluster->y());

  if (cp.theCluster->getClusterErrorX() < 0.) {
    return dp.theTopol->localPosition(pos);
  } else {
    LocalPoint out(cp.theCluster->getClusterPosX(), dp.theTopol->localY(pos.y()));
    return out;
  }
}

LocalError MTDCPEBase::localError(DetParam const& dp, ClusterParam& cp) const {
  MeasurementPoint pos(cp.theCluster->x(), cp.theCluster->y());
  float sigma2 = cp.theCluster->positionError(sigma_flat);
  sigma2 *= sigma2;
  MeasurementError posErr(sigma2, 0, sigma2);
  LocalError outErr = dp.theTopol->localError(pos, posErr);

  if (cp.theCluster->getClusterErrorX() < 0.) {
    return outErr;
  } else {
    LocalPoint outPos(cp.theCluster->getClusterPosX(), dp.theTopol->localY(pos.y()));
    float sigmaX2 = cp.theCluster->getClusterErrorX();
    sigmaX2 *= sigmaX2;
    LocalError outErrDet(sigmaX2, 0, outErr.yy());
    return outErrDet;
  }
}

MTDCPEBase::TimeValue MTDCPEBase::clusterTime(DetParam const& dp, ClusterParam& cp) const {
  return cp.theCluster->time();
}

MTDCPEBase::TimeValueError MTDCPEBase::clusterTimeError(DetParam const& dp, ClusterParam& cp) const {
  return cp.theCluster->timeError();
}
