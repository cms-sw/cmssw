#ifndef Geometry_TrackerGeometryBuilder_StripGeomDetType_H
#define Geometry_TrackerGeometryBuilder_StripGeomDetType_H

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <vector>

/**
 * StripGeomDetType is the abstract class for SiStripGeomDetType.
 */
class StripGeomDetType : public GeomDetType
{

public:

  typedef  StripTopology        TopologyType;

  StripGeomDetType(TopologyType* t, std::string const & name,SubDetector& det,bool stereo) : GeomDetType(name,det),
    theTopology(t),theStereoFlag(stereo){}

  virtual ~StripGeomDetType() {
    delete theTopology;
  }

  // Access to topologies
  virtual const Topology&  topology() const;
  
  virtual const  TopologyType& specificTopology() const;

  void setTopology( TopologyType* topol);

  bool isStereo() const {return theStereoFlag;}

private:

  TopologyType*    theTopology;
  bool           theStereoFlag;

};

#endif // StripGeomDetType_H
