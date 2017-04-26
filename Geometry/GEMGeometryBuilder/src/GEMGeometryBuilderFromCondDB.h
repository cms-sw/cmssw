#ifndef Geometry_GEMGeometry_GEMGeometryBuilderFromCondDB_H
#define Geometry_GEMGeometry_GEMGeometryBuilderFromCondDB_H

/** \class  GEMGeometryBuilderFromCondDB
 *  Build the GEMGeometry from the RecoIdealGeometry description stored in Condition DB 
 *
 *  \author M. Maggi - INFN Bari
 *
 */

#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include <string>
#include <map>
#include <list>

class GEMGeometry;
class GEMDetId;
class GEMEtaPartition;
class GEMSuperChamber;
class GEMChamber;

class GEMGeometryBuilderFromCondDB 
{
 public:

  GEMGeometryBuilderFromCondDB();

  ~GEMGeometryBuilderFromCondDB();
  
  GEMGeometry* build(const RecoIdealGeometry& rgeo);
  
 private:
  typedef ReferenceCountingPointer<Plane> RCPPlane;
  
  GEMSuperChamber* buildSuperChamber( const GEMDetId id, const RecoIdealGeometry& rig,
				      size_t idt ) const;
  
  GEMChamber* buildChamber( const GEMDetId id,  const RecoIdealGeometry& rig,
			    size_t idt ) const;
  
  std::map<GEMDetId, std::list<GEMEtaPartition *> > chids;
};

#endif
