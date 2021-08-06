#ifndef RPCGeometry_RPCGeometryParsFromDD_H
#define RPCGeometry_RPCGeometryParsFromDD_H

/* \class  RPCGeometryParsFromDD
 *  Build the RPCGeometry from the DDD and DD4Hep description
 *  
 *  DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
 *  Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) 
 *  Created:  Mon, 09 Nov 2020 
 *
 */

#include <string>
#include <map>
#include <list>

class DDCompactView;
class DDFilteredView;
namespace cms {  // DD4Hep
  class DDFilteredView;
  class DDCompactView;
}  // namespace cms
class RPCDetId;
class RPCRoll;
class MuonGeometryConstants;
class RecoIdealGeometry;
class RPCGeometryParsFromDD {
public:
  RPCGeometryParsFromDD();

  ~RPCGeometryParsFromDD();

  // DD
  void build(const DDCompactView* cview, const MuonGeometryConstants& muonConstants, RecoIdealGeometry& rgeo);
  // DD4Hep
  void build(const cms::DDCompactView* cview, const MuonGeometryConstants& muonConstants, RecoIdealGeometry& rgeo);

private:
  // DD
  void buildGeometry(DDFilteredView& fview, const MuonGeometryConstants& muonConstants, RecoIdealGeometry& rgeo);
  // DD4Hep
  void buildGeometry(cms::DDFilteredView& fview, const MuonGeometryConstants& muonConstants, RecoIdealGeometry& rgeo);
};

#endif
