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
//#include <vector>


class DDCompactView;
class DDFilteredView;
class RPCGeometry;
//class RPCChamber;

class RPCGeometryBuilderFromDDD 
{ 
 public:

  RPCGeometryBuilderFromDDD(bool comp11);

  ~RPCGeometryBuilderFromDDD();

  RPCGeometry* build(const DDCompactView* cview);


 private:
  RPCGeometry* buildGeometry(DDFilteredView& fview);
  
  //  RPCChamber* buildChamber(DDFileterView& fview,
  //			   RPCGeometry& geometry,
  //			   const std::string& type);

  //std::vector<double> extractParameters(DDFilteredView& fview);
  bool theComp11Flag;

};

#endif
