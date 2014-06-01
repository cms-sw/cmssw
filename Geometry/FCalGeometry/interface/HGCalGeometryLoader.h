#ifndef GeometryFcalGeometryHGCalGeometryLoader_h
#define GeometryFcalGeometryHGCalGeometryLoader_h

class HGCalTopology;
class HGCalGeometry;

class HGCalGeometryLoader {

public:
  HGCalGeometryLoader ();
  ~HGCalGeometryLoader ();

  HGCalGeometry* build(const HGCalTopology& );
};

#endif
