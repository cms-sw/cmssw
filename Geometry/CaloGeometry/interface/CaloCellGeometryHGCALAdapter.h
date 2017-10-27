#ifndef GeometryCaloCellGeometryHGCALAdapter
#define GeometryCaloCellGeometryHGCALAdapter

#include "Geometry/CaloGeometry/interface/FlatTrd.h"

class CaloCellGeometryHGCALAdapter : public FlatTrd {
public:
  explicit CaloCellGeometryHGCALAdapter (const FlatTrd * f, GlobalPoint p) : FlatTrd(*f), position_(p) {}
  CaloCellGeometryHGCALAdapter(const CaloCellGeometryHGCALAdapter&) = delete;
  CaloCellGeometryHGCALAdapter operator=(const CaloCellGeometryHGCALAdapter &) = delete;
  const GlobalPoint& getPosition() const override {return position_;}
private:
  GlobalPoint position_;
};

#endif
