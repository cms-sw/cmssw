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
//         Modified: Thu, 04 June 2020, following what made in PR #30047               
//   
//         Original author: Tim Cox
*/
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <string>

class DDCompactView;
class CSCGeometry;
class MuonGeometryConstants;
namespace cms {
  class DDFilteredView;
  class DDCompactView;
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
  void build(CSCGeometry& geom, const cms::DDCompactView* cview, const MuonGeometryConstants& muonConstants);

protected:
private:
  const std::string myName;
};
#endif
