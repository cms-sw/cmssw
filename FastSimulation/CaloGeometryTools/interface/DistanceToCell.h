#ifndef FastSimulation_CalorimeterProperties_DistanceToCell_h
#define FastSimulation_CalorimeterProperties_DistanceToCell_h
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/DetId/interface/DetId.h"

// used in GeometryHelper to sort the vector given by GetWindow
// not much optimized.

class CaloSubdetectorGeometry;

class DistanceToCell {
public:
  DistanceToCell();
  DistanceToCell(const DistanceToCell&);
  DistanceToCell(const CaloSubdetectorGeometry* det, const DetId& cell);
  ~DistanceToCell() { ; }
  bool operator()(const DetId& c1, const DetId& c2);

private:
  const CaloSubdetectorGeometry* det_;
  DetId pivot_;
  GlobalPoint pivotPosition_;
};

#endif
