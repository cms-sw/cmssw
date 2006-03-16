#ifndef Geometry_TrackerGeometryBuilder_PixelGeomDetType_H
#define Geometry_TrackerGeometryBuilder_PixelGeomDetType_H


#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

/**
 * Generic DetType for the Pixels. Specialized in SiPixelGeomDetType.
 */

class PixelGeomDetType : public GeomDetType {

public:
  typedef  PixelTopology        TopologyType;

  PixelGeomDetType(TopologyType* t,std::string& name, SubDetector& det);

  virtual ~PixelGeomDetType() {
    delete theTopology;
  }

  // Access to topologies
  virtual const  Topology& topology() const;

  virtual const TopologyType& specificTopology() const;


  void setTopology( TopologyType* topol);

private:

  TopologyType*    theTopology;

};



#endif // PixelGeomDetType_H
