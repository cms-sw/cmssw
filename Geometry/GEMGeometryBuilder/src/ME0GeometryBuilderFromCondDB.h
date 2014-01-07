#ifndef ME0Geometry_ME0GeometryBuilderFromCondDB_H
#define ME0Geometry_ME0GeometryBuilderFromCondDB_H

/** \class  ME0GeometryBuilderFromCondDB
 *  Build the ME0Geometry from the DDD description stored in Condition DB 
 *
 *  \author M. Maggi - INFN Bari
 *
 */

#include <CondFormats/GeometryObjects/interface/RecoIdealGeometry.h>
#include <string>
#include <map>
#include <list>


class ME0Geometry;
class ME0DetId;
class ME0EtaPartition;

class ME0GeometryBuilderFromCondDB 
{ 
 public:

  ME0GeometryBuilderFromCondDB(bool comp11);

  ~ME0GeometryBuilderFromCondDB();

  ME0Geometry* build(const RecoIdealGeometry& rgeo);


 private:
  //  std::map<ME0DetId,std::list<ME0EtaPartition *> > chids;
  bool theComp11Flag;

};

#endif
