#ifndef RPCGeometry_RPCGeometryParsFromDD_H
#define RPCGeometry_RPCGeometryParsFromDD_H

/** \class  RPCGeometryParsFromDD
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
class RPCDetId;
class RPCRoll;
class MuonDDDConstants;
class RecoIdealGeometry;
class RPCGeometryParsFromDD 
{ 
 public:

  RPCGeometryParsFromDD();

  ~RPCGeometryParsFromDD();

  void build(const DDCompactView* cview, 
	     const MuonDDDConstants& muonConstants,
	     RecoIdealGeometry& rgeo);


 private:
  void buildGeometry(DDFilteredView& fview, 
		     const MuonDDDConstants& muonConstants,
		     RecoIdealGeometry& rgeo);


};

#endif
