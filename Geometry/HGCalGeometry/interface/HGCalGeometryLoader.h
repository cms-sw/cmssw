#ifndef GeometryHGCalGeometryHGCalGeometryLoader_h
#define GeometryHGCalGeometryHGCalGeometryLoader_h

class HGCalTopology;
class HGCalGeometry;

class HGCalGeometryLoader {

public:
  HGCalGeometryLoader ();
  ~HGCalGeometryLoader ();

  HGCalGeometry* build(const HGCalTopology& );
};

#endif
