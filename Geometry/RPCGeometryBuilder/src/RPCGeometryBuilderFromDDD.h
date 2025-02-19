#ifndef RPCGeometry_RPCGeometryBuilderFromDDD_H
#define RPCGeometry_RPCGeometryBuilderFromDDD_H

/** \class  RPCGeometryBuilderFromDDD
 *  Build the RPCGeometry ftom the DDD description
 *
 *  \author Port of: MuDDDRPCBuilder, MuonRPCGeometryBuilder (ORCA)
 *  \author M. Maggi - INFN Bari
 *
 */

#include <string>
#include <map>
#include <list>

class DDCompactView;
class DDFilteredView;
class RPCGeometry;
class RPCDetId;
class RPCRoll;
class MuonDDDConstants;

class RPCGeometryBuilderFromDDD 
{ 
 public:

  RPCGeometryBuilderFromDDD(bool comp11);

  ~RPCGeometryBuilderFromDDD();

  RPCGeometry* build(const DDCompactView* cview, const MuonDDDConstants& muonConstants);


 private:
  RPCGeometry* buildGeometry(DDFilteredView& fview, const MuonDDDConstants& muonConstants);
  std::map<RPCDetId,std::list<RPCRoll *> > chids;

  bool theComp11Flag;

};

#endif
