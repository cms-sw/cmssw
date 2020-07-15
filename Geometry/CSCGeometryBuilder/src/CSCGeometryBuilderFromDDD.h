#ifndef CSCGeometryBuilder_CSCGeometryBuilderFromDDD_h
#define CSCGeometryBuilder_CSCGeometryBuilderFromDDD_h

/*
// \class CSCGeometryBuilderFromDDD
//
//  Description: CSC Geometry Builder for DD4hep
//              
//
// \author Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//         Created:  Thu, 05 March 2020 
//   
//         Original author: Tim Cox
*/
//

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include <string>

class DDCompactView;
class CSCGeometry;
class MuonGeometryConstants;
namespace cms {
  class DDFilteredView;
  class DDCompactView;
  class MuonNumbering;
}  // namespace cms

class CSCGeometryBuilderFromDDD {
public:
  /// Constructor
  CSCGeometryBuilderFromDDD();

  /// Destructor
  virtual ~CSCGeometryBuilderFromDDD();

  // Build the geometry DDD
  void build(CSCGeometry& geom, const DDCompactView* fv, const MuonGeometryConstants& muonConstants);

  // Build the geometry dd4hep
  void build(CSCGeometry& geom, const cms::DDCompactView* cview, const cms::MuonNumbering& muonConstants);

protected:
private:
  const std::string myName;
};
#endif
