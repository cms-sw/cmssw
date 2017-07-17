#ifndef Geometry_GEMGeometry_ME0GeometryBuilderFromDDD_H
#define Geometry_GEMGeometry_ME0GeometryBuilderFromDDD_H

/** \class  ME0GeometryBuilderFromDDD
 *  Build the ME0Geometry ftom the DDD description
 *
 *  \author M. Maggi - INFN Bari
 *
 */

#include <string>
#include <map>
#include <vector>

class DDCompactView;
class DDFilteredView;
class ME0Geometry;
class ME0DetId;
class ME0EtaPartition;
class MuonDDDConstants;

class ME0GeometryBuilderFromDDD 
{ 
 public:

  ME0GeometryBuilderFromDDD();

  ~ME0GeometryBuilderFromDDD();

  ME0Geometry* build(const DDCompactView* cview, const MuonDDDConstants& muonConstants);

 private:
  ME0Geometry* buildGeometry(DDFilteredView& fview, const MuonDDDConstants& muonConstants);
  std::map<ME0DetId,std::vector<ME0DetId>> chids;
};

#endif
