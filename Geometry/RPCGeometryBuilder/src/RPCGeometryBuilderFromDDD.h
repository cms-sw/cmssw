/** \class  RPCGeometryBuilderFromDDD
 *  Build the RPCGeometry ftom the DDD description
 *
 *  \author Port of: MuDDDRPCBuilder, MuonRPCGeometryBuilder (ORCA)
 *  \author M. Maggi - INFN Bari
 *
 */

#ifndef RPCSimAlgo_RPCGeometryBuilderFromDDD_H
#define RPCSimAlgo_RPCGeometryBuilderFromDDD_H


#include <string>
//#include <vector>


class DDCompactView;
class DDFilteredView;
class RPCGeometry;
//class RPCChamber;

class RPCGeometryBuilderFromDDD 
{ 
 public:

  RPCGeometryBuilderFromDDD();

  ~RPCGeometryBuilderFromDDD();

  RPCGeometry* build(const DDCompactView* cview);


 private:
  RPCGeometry* buildGeometry(DDFilteredView& fview);
  
  //  RPCChamber* buildChamber(DDFileterView& fview,
  //			   RPCGeometry& geometry,
  //			   const std::string& type);

  //std::vector<double> extractParameters(DDFilteredView& fview);

};

#endif
