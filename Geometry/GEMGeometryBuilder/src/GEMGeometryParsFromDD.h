#ifndef GEMGeometry_GEMGeometryParsFromDD_H
#define GEMGeometry_GEMGeometryParsFromDD_H

/** \class  GEMGeometryParsFromDD
 *  Build the GEMGeometry ftom the DDD description
 *
 *  \author M. Maggi - INFN Bari
 *
 */

#include <string>
#include <map>
#include <list>

class DDCompactView;
class DDFilteredView;
class MuonDDDConstants;
class RecoIdealGeometry;
class GEMGeometryParsFromDD 
{ 
 public:

  GEMGeometryParsFromDD();

  ~GEMGeometryParsFromDD();

  void build(const DDCompactView* cview, 
	     const MuonDDDConstants& muonConstants,
	     RecoIdealGeometry& rgeo);


 private:
  void buildGeometry(DDFilteredView& fview, 
		     const MuonDDDConstants& muonConstants,
		     RecoIdealGeometry& rgeo);


};

#endif
