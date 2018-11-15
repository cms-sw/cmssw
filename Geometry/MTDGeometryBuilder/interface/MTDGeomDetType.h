#ifndef Geometry_MTDGeometryBuilder_MTDGeomDetType_H
#define Geometry_MTDGeometryBuilder_MTDGeomDetType_H

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

/**
 * Generic DetType for the MTD. 
 */

class MTDGeomDetType final: public GeomDetType {

public:
  using TopologyType = PixelTopology;

  MTDGeomDetType(TopologyType* t,std::string const& name, SubDetector& det) :
    GeomDetType(name,det),
    theTopology(t){}

  ~MTDGeomDetType() override {
  }

  // Access to topologies
  const  Topology& topology() const override { return *theTopology;}

  virtual const TopologyType& specificTopology() const  { return *theTopology;}

  MTDGeomDetType& operator = ( const MTDGeomDetType& other ) = delete;
  MTDGeomDetType( const MTDGeomDetType& other ) = delete;

 private:    
  // take ownership of the topology when passed through constructor
  std::unique_ptr<TopologyType>    theTopology;
};



#endif // PixelGeomDetType_H
