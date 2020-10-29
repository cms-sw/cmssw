#ifndef RecoMTD_DetLayers_MTDDiskSectorBuilderFromDet_H
#define RecoMTD_DetLayers_MTDDiskSectorBuilderFromDet_H

#include "DataFormats/GeometrySurface/interface/BoundDiskSector.h"
#include "DataFormats/GeometrySurface/interface/DiskSectorBounds.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <utility>
#include <vector>
#include <iostream>

/** The trapezoid has the minimal size fully containing all Dets.
 */

class MTDDiskSectorBuilderFromDet {
public:
  BoundDiskSector* operator()(const std::vector<const GeomDet*>& dets) const;
};

#endif
