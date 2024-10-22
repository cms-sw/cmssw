#ifndef Geometry_TrackerNumberingBuilder_TrackerShapeToBounds_H
#define Geometry_TrackerNumberingBuilder_TrackerShapeToBounds_H

#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"
#include <vector>
#include <algorithm>
/**
 * Converts DDSolid volumes to Bounds
 */
class TrackerShapeToBounds {
public:
  /**
   *buildBounds() return the Bounds.
   */
  Bounds* buildBounds(const cms::DDSolidShape&, const std::vector<double>&) const;

private:
  Bounds* buildBox(const std::vector<double>&) const;
  Bounds* buildTrap(const std::vector<double>&) const;
  Bounds* buildOpen() const;
};

#endif
