#ifndef RPCGeometry_RPCGeometryBuilderFromCondDB_H
#define RPCGeometry_RPCGeometryBuilderFromCondDB_H

/** \class  RPCGeometryBuilderFromCondDB
 *  Build the RPCGeometry from the DDD description stored in Condition DB 
 *
 *  \author M. Maggi - INFN Bari
 *
 */

#include <CondFormats/GeometryObjects/interface/RecoIdealGeometry.h>
#include <string>
#include <map>
#include <memory>
#include <list>


class RPCGeometry;
class RPCDetId;
class RPCRoll;

class RPCGeometryBuilderFromCondDB 
{ 
 public:

  RPCGeometryBuilderFromCondDB(bool comp11);

  ~RPCGeometryBuilderFromCondDB();

  RPCGeometry* build(const RecoIdealGeometry& rgeo);


 private:
  std::map<RPCDetId,std::list< std::shared_ptr< RPCRoll >>> chids;
  bool theComp11Flag;

};

#endif
