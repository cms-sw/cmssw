#ifndef RPCGeometry_RPCGeometryBuilderFromDDD_H
#define RPCGeometry_RPCGeometryBuilderFromDDD_H

/** 
 *
 */
#include "Geometry/MuonNumbering/interface/DD4hep_RPCNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include <string>
#include <map>
#include <list>
#include <memory>

class DDCompactView;
class DDFilteredView;
namespace cms {
  class DDFilteredView;
  class DDCompactView;
  class MuonNumbering;
  struct DDSpecPar;
  struct DDSpecParRegistry;
}
class RPCGeometry;
class RPCDetId;
class RPCRoll;
class MuonDDDConstants;

class RPCGeometryBuilderFromDDD {
 public:
  RPCGeometryBuilderFromDDD(bool comp11);
  
  ~RPCGeometryBuilderFromDDD();
  
  // for DDD
  RPCGeometry* build(const DDCompactView* cview, const MuonDDDConstants& muonConstants);
  // for DD4hep
  RPCGeometry* build(const cms::DDCompactView* cview, const cms::MuonNumbering& muonConstants);


  
 private:
  // for DDD
  RPCGeometry* buildGeometry(DDFilteredView& fview, const MuonDDDConstants& muonConstants);
  // for DD4hep
  RPCGeometry* buildGeometry(cms::DDFilteredView& fview, const cms::MuonNumbering& muonConstants);
  

  std::map<RPCDetId, std::list<RPCRoll*> > chids;

  std::unique_ptr<cms::RPCNumberingScheme> rpcnum_ = nullptr;  
  bool theComp11Flag;
};



#endif
