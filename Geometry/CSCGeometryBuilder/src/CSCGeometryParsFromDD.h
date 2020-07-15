#ifndef CSCGeometryBuilder_CSCGeometryParsFromDD_h
#define CSCGeometryBuilder_CSCGeometryParsFromDD_h

/*
// \class CSCGeometryParsFromDDD
//
//  Description: CSC Geometry Pars for DD4hep
//              
//
// \author Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//         Created:  Thu, 05 March 2020 
//   
//         Original author: Tim Cox
*/

#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <string>

class CSCGeometry;
class DDCompactView;
class MuonGeometryConstants;
class RecoIdealGeometry;
class CSCRecoDigiParameters;

namespace cms {
  class DDFilteredView;
  class DDCompactView;
  class MuonNumbering;
}  // namespace cms

class CSCGeometryParsFromDD {
public:
  /// Constructor
  CSCGeometryParsFromDD();

  /// Destructor
  virtual ~CSCGeometryParsFromDD();

  /// Build the geometry returning the RecoIdealGeometry and the CSCRecoDigiParameters objects
  // as built from the DDD.
  bool build(const DDCompactView* cview,
             const MuonGeometryConstants& muonConstants,
             RecoIdealGeometry& rig,
             CSCRecoDigiParameters& rdp);
  //dd4hep
  bool build(const cms::DDCompactView* cview,
             const cms::MuonNumbering& muonConstants,
             RecoIdealGeometry& rig,
             CSCRecoDigiParameters& rdp);

private:
  std::string myName;
};
#endif
