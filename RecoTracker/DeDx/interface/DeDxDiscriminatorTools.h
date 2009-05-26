#ifndef DeDxDiscriminatorTools_H
#define DeDxDiscriminatorTools_H


#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"



namespace DeDxDiscriminatorTools
{

  //   using namespace std;

   bool   IsSpanningOver2APV   (unsigned int FirstStrip, unsigned int ClusterSize);
  bool   IsSaturatingStrip    (const std::vector<uint8_t>& Ampls);
  double charge               (const std::vector<uint8_t>& Ampls);
   double path                 (double cosine, double thickness);

   bool   IsFarFromBorder      (TrajectoryStateOnSurface trajState, const GeomDetUnit* it);

}
#endif
