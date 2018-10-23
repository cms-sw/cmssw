#ifndef Geometry_GEMGeometry_GEMGeometryBuilderFromCondDB_H
#define Geometry_GEMGeometry_GEMGeometryBuilderFromCondDB_H

/** \class  GEMGeometryBuilderFromCondDB
 *  Build the GEMGeometry from the RecoIdealGeometry description stored in Condition DB 
 *
 *  \author M. Maggi - INFN Bari
 *
 */

#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

class GEMGeometryBuilderFromCondDB 
{
 public:

  GEMGeometryBuilderFromCondDB();

  ~GEMGeometryBuilderFromCondDB();
  
  void build(GEMGeometry& theGeometry,
	     const RecoIdealGeometry& rgeo );
  
 private:
  typedef ReferenceCountingPointer<BoundPlane> RCPBoundPlane;

  GEMSuperChamber* buildSuperChamber(const RecoIdealGeometry& rgeo, unsigned int gid, GEMDetId detId) const;
  
  GEMChamber* buildChamber(const RecoIdealGeometry& rgeo, unsigned int gid, GEMDetId detId) const;

  GEMEtaPartition* buildEtaPartition(const RecoIdealGeometry& rgeo, unsigned int gid, GEMDetId detId) const;

  RCPBoundPlane boundPlane(const RecoIdealGeometry& rgeo, unsigned int gid, GEMDetId detId) const;
  
};

#endif
