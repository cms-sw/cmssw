#ifndef RPCGeometry_RPCGeometryBuilderFromDDD_H
#define RPCGeometry_RPCGeometryBuilderFromDDD_H

/** \class  RPCGeometryBuilderFromDDD
 *  Build the RPCGeometry ftom the DDD description
 *
 *  \author Port of: MuDDDRPCBuilder, MuonRPCGeometryBuilder (ORCA)
 *  \author M. Maggi - INFN Bari
 *
 */

#include "Geometry/Surface/interface/BoundPlane.h"

#include <string>
#include <vector>


class DDCompactView;
class DDFilteredView;
class RPCGeometry;
class RPCChamber;
class RPCRoll;
class Bounds;

class RPCGeometryBuilderFromDDD 
{ 
 public:

  RPCGeometryBuilderFromDDD();

  ~RPCGeometryBuilderFromDDD();

  RPCGeometry* build(const DDCompactView* cview);


 private:

  RPCChamber* buildChamber(DDFilteredView& fview) const;
  
  RPCRoll* buildRoll(DDFilteredView& fview,
		     RPCChamber* ch) const;

  /// get parameter also for boolean solid.
  std::vector<double> extractParameters(DDFilteredView& fview) const ;
  
/*   typedef ReferenceCountingPointer<BoundPlane> RCPPlane; */

/*   RCPPlane plane(const DDFilteredView& fview,  */
/* 		 const Bounds& bound) const; */

  RPCGeometry* buildGeometry(DDFilteredView& fview) const;

};

#endif
