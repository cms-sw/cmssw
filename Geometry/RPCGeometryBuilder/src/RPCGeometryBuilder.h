#ifndef RPCGeometry_RPCGeometryBuilder_H
#define RPCGeometry_RPCGeometryBuilder_H
/*
//\class RPCGeometryBuilder

 Description: RPC Geometry builder from DD & DD4hep
              DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osborne made for DTs (DD4HEP migration)
//          Created:  Fri, 20 Sep 2019 
//          Modified: Fri, 29 May 2020, following what Sunanda Banerjee made in PR #29842 PR #29943 and Ianna Osborne in PR #29954    
*/
#include <string>
#include <map>
#include <list>
#include <memory>
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

class DDCompactView;
class DDFilteredView;
namespace cms {
  class DDFilteredView;
  class DDCompactView;
}  // namespace cms
class RPCGeometry;
class RPCRoll;
class MuonGeometryConstants;

class RPCGeometryBuilder {
public:
  RPCGeometryBuilder();

  // for DDD
  std::unique_ptr<RPCGeometry> build(const DDCompactView* cview, const MuonGeometryConstants& muonConstants);
  // for DD4hep
  std::unique_ptr<RPCGeometry> build(const cms::DDCompactView* cview, const MuonGeometryConstants& muonConstants);

private:
  // for DDD
  std::unique_ptr<RPCGeometry> buildGeometry(DDFilteredView& fview, const MuonGeometryConstants& muonConstants);
  // for DD4hep
  std::unique_ptr<RPCGeometry> buildGeometry(cms::DDFilteredView& fview, const MuonGeometryConstants& muonConstants);
  std::map<RPCDetId, std::list<RPCRoll*> > chids;
};

#endif
