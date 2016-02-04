

#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

float BoundDisk::innerRadius() const {
  return dynamic_cast<const SimpleDiskBounds&>(bounds()).innerRadius();
}

float BoundDisk::outerRadius() const {
  return dynamic_cast<const SimpleDiskBounds&>(bounds()).outerRadius();
}
