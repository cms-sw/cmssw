#ifndef CSCGeometryBuilder_CSCGeometryParsFromDD_h
#define CSCGeometryBuilder_CSCGeometryParsFromDD_h

/** \class CSCGeometryParsFromDD
 *
 *  Build the CSCGeometry from the DDD description.
 *
 *  \author Tim Cox
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <string>
#include <boost/shared_ptr.hpp>

class CSCGeometry;
class DDCompactView;
class MuonDDDConstants;
class RecoIdealGeometry;
class CSCRecoDigiParameters;

class CSCGeometryParsFromDD {
 public:

  /// Constructor
  CSCGeometryParsFromDD( );
  
  /// Destructor
  virtual ~CSCGeometryParsFromDD();

  /// Build the geometry returning the RecoIdealGeometry and the CSCRecoDigiParameters objects
  // as built from the DDD.
  bool build( const DDCompactView* cview 
	      , const MuonDDDConstants& muonConstants
	      , RecoIdealGeometry& rig
	      , CSCRecoDigiParameters& rdp
	      );

 private:
  std::string myName;

};
#endif

