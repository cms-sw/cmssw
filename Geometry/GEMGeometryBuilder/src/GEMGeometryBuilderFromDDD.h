#ifndef GEMGeometry_GEMGeometryBuilderFromDDD_H
#define GEMGeometry_GEMGeometryBuilderFromDDD_H

/** \class  GEMGeometryBuilderFromDDD
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
class GEMGeometry;
class GEMDetId;
class GEMEtaPartition;
class MuonDDDConstants;

class GEMGeometryBuilderFromDDD 
{ 
 public:

  GEMGeometryBuilderFromDDD(bool comp11);

  ~GEMGeometryBuilderFromDDD();

  GEMGeometry* build(const DDCompactView* cview, const MuonDDDConstants& muonConstants);


 private:
  GEMGeometry* buildGeometry(DDFilteredView& fview, const MuonDDDConstants& muonConstants);
  //  std::map<GEMDetId,std::list<GEMEtaPartition *> > chids;

  bool theComp11Flag;

};

#endif
