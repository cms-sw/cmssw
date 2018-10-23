#ifndef Geometry_GEMGeometry_GEMGeometryBuilderFromDDD_H
#define Geometry_GEMGeometry_GEMGeometryBuilderFromDDD_H

#include "DataFormats/GeometrySurface/interface/Plane.h"
/** \class  GEMGeometryBuilderFromDDD
 *  Build the GEMGeometry ftom the DDD description
 *
 *  \author M. Maggi - INFN Bari
 *
 */

#include <string>
#include <map>
#include <vector>

class DDCompactView;
class DDFilteredView;
class GEMGeometry;
class GEMDetId;
class GEMSuperChamber;
class GEMChamber;
class GEMEtaPartition;
class MuonDDDConstants;

class GEMGeometryBuilderFromDDD 
{ 
 public:

  GEMGeometryBuilderFromDDD();

  ~GEMGeometryBuilderFromDDD();

  void build(GEMGeometry& theGeometry,
	     const DDCompactView* cview, const MuonDDDConstants& muonConstants);
  
 private:
  std::map<GEMDetId,std::vector<GEMDetId>> chids;

  typedef ReferenceCountingPointer<BoundPlane> RCPBoundPlane;
  
  RCPBoundPlane boundPlane(const DDFilteredView& fv,
			   Bounds* bounds, bool isOddChamber) const ;
  
  GEMSuperChamber* buildSuperChamber(DDFilteredView& fv, GEMDetId detId) const;

  GEMChamber* buildChamber(DDFilteredView& fv, GEMDetId detId) const;
  
  GEMEtaPartition* buildEtaPartition(DDFilteredView& fv, GEMDetId detId) const;
};

#endif
