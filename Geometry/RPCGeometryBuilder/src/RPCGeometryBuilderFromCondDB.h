#ifndef RPCGeometry_RPCGeometryBuilderFromCondDB_H
#define RPCGeometry_RPCGeometryBuilderFromCondDB_H

/** \class  RPCGeometryBuilderFromCondDB
 *  Build the RPCGeometry from the DDD description stored in Condition DB 
 *
 *  \author M. Maggi - INFN Bari
 *
 */

#include <CondFormats/GeometryObjects/interface/RecoIdealGeometry.h>
#include "DataFormats/MuonDetId/interface/RPCDetIdFwd.h"
#include <string>
#include <map>
#include <list>

class RPCGeometry;
class RPCRoll;

class RPCGeometryBuilderFromCondDB {
public:
  RPCGeometryBuilderFromCondDB();

  ~RPCGeometryBuilderFromCondDB();

  RPCGeometry* build(const RecoIdealGeometry& rgeo);

private:
  std::map<RPCDetId, std::list<RPCRoll*> > chids;
};

#endif
