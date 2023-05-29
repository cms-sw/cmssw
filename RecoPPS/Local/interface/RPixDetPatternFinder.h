/*
 *
* This is a part of CTPPS offline software.
* Author:
*   Fabrizio Ferro (ferro@ge.infn.it)
*   Enrico Robutti (robutti@ge.infn.it)
*   Fabio Ravera   (fabio.ravera@cern.ch)
*
*/
#ifndef RecoPPS_Local_RPixDetPatternFinder_H
#define RecoPPS_Local_RPixDetPatternFinder_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "DataFormats/Math/interface/Error.h"

#include <vector>

class RPixDetPatternFinder {
public:
  RPixDetPatternFinder(edm::ParameterSet const &parameterSet) {}

  virtual ~RPixDetPatternFinder(){};

  typedef struct {
    CTPPSGeometry::Vector globalPoint;
    math::Error<3>::type globalError;
    CTPPSPixelRecHit recHit;
    CTPPSPixelDetId detId;
  } PointInPlane;
  typedef std::vector<PointInPlane> Road;

  void setHits(const edm::DetSetVector<CTPPSPixelRecHit> *hitVector) { hitVector_ = hitVector; }
  virtual void findPattern(bool *is2planepot) = 0;
  void clear() { patternVector_.clear(); }
  std::vector<Road> const &getPatterns() const { return patternVector_; }
  void setGeometry(const CTPPSGeometry *geometry) { geometry_ = geometry; }

protected:
  const edm::DetSetVector<CTPPSPixelRecHit> *hitVector_;
  std::vector<Road> patternVector_;
  const CTPPSGeometry *geometry_;
};

#endif
