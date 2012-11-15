#ifndef GEMGeometry_GEMGeometryBuilderFromCondDB_H
#define GEMGeometry_GEMGeometryBuilderFromCondDB_H

/** \class  GEMGeometryBuilderFromCondDB
 *  Build the GEMGeometry from the DDD description stored in Condition DB 
 *
 *  \author M. Maggi - INFN Bari
 *
 */

#include <CondFormats/GeometryObjects/interface/RecoIdealGeometry.h>
#include <string>
#include <map>
#include <list>


class GEMGeometry;
class GEMDetId;
class GEMEtaPartition;

class GEMGeometryBuilderFromCondDB 
{ 
 public:

  GEMGeometryBuilderFromCondDB(bool comp11);

  ~GEMGeometryBuilderFromCondDB();

  GEMGeometry* build(const RecoIdealGeometry& rgeo);


 private:
  //  std::map<GEMDetId,std::list<GEMEtaPartition *> > chids;
  bool theComp11Flag;

};

#endif
