#ifndef CSCGeometryBuilder_CSCGeometryBuilder_h
#define CSCGeometryBuilder_CSCGeometryBuilder_h

/** \class CSCGeometryBuilder
 *
 *  Build the CSCGeometry from the DDD description.
 *
 *  \author Tim Cox
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <Geometry/CSCGeometry/src/CSCWireGroupPackage.h>
#include <CondFormats/GeometryObjects/interface/CSCRecoDigiParameters.h>
#include <CondFormats/GeometryObjects/interface/RecoIdealGeometry.h>

#include <string>

#include <memory>

class CSCGeometry;

class CSCGeometryBuilder {
public:
  /// Constructor
  CSCGeometryBuilder();

  /// Destructor
  virtual ~CSCGeometryBuilder();

  /// Build the geometry
  void build( const std::shared_ptr<CSCGeometry>& theGeometry
		      , const RecoIdealGeometry& rig
		      , const CSCRecoDigiParameters& cscpars ) ;

protected:

private:
  /// Build one CSC chamber, and its component layers, and add them to the geometry
  void buildChamber (  
		     const std::shared_ptr<CSCGeometry>& theGeometry        // the geometry container
		     , CSCDetId chamberId              // the DetId of this chamber
		     , const std::vector<float>& fpar  // volume parameters
		     , const std::vector<float>& fupar // user parameters
		     , const std::vector<float>& gtran // translation vector
		     , const std::vector<float>& grmat // rotation matrix
		     , const CSCWireGroupPackage& wg   // wire group info
	);

  const std::string myName;

};
#endif

