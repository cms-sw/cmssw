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
//         Modified: Thu, 04 June 2020, following what made in PR #30047               
//   
//         Original author: Tim Cox
*/
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
             const MuonGeometryConstants& muonConstants,
             RecoIdealGeometry& rig,
             CSCRecoDigiParameters& rdp);

private:
  std::string myName;
};
#endif
