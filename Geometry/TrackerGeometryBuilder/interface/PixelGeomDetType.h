#ifndef Geometry_TrackerGeometryBuilder_PixelGeomDetType_H
#define Geometry_TrackerGeometryBuilder_PixelGeomDetType_H

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

/**
 * Generic DetType for the Pixels. Specialized in SiPixelGeomDetType.
 */

class PixelGeomDetType final: public GeomDetType {

public:
  using TopologyType = PixelTopology;

  PixelGeomDetType(TopologyType* t,std::string const& name, SubDetector& det) :
    GeomDetType(name,det),
    theTopology(t){}

  ~PixelGeomDetType() override {
    delete theTopology;
  }

  // Access to topologies
  const  Topology& topology() const override { return *theTopology;}

  virtual const TopologyType& specificTopology() const  { return *theTopology;}

  PixelGeomDetType& operator = ( const PixelGeomDetType& other ) = delete;
  PixelGeomDetType( const PixelGeomDetType& other ) = delete;

 private:    
  TopologyType*    theTopology;
};



#endif // PixelGeomDetType_H
