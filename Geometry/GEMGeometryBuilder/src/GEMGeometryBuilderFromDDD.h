#ifndef Geometry_GEMGeometry_GEMGeometryBuilderFromDDD_H
#define Geometry_GEMGeometry_GEMGeometryBuilderFromDDD_H

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
class GEMEtaPartition;
class MuonDDDConstants;

class GEMGeometryBuilderFromDDD 
{ 
 public:

  GEMGeometryBuilderFromDDD();

  ~GEMGeometryBuilderFromDDD();

  GEMGeometry* build(const DDCompactView* cview, const MuonDDDConstants& muonConstants);

 private:
  GEMGeometry* buildGeometry(DDFilteredView& fview, const MuonDDDConstants& muonConstants);
  std::map<GEMDetId,std::vector<GEMDetId>> chids;
};

#endif
