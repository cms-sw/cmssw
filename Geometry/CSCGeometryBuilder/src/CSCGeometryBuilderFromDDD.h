#ifndef CSCGeometryBuilder_CSCGeometryBuilderFromDDD_h
#define CSCGeometryBuilder_CSCGeometryBuilderFromDDD_h

/** \class CSCGeometryBuilderFromDDD
 *
 *  Build the CSCGeometry from the DDD description.
 *
 *  \author Tim Cox
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <boost/shared_ptr.hpp>
#include <string>

class DDCompactView;
class DDFilteredView;
class CSCGeometry;
class CSCWireGroupPackage;
class MuonDDDConstants;

class CSCGeometryBuilderFromDDD {
public:
  /// Constructor
  CSCGeometryBuilderFromDDD();

  /// Destructor
  virtual ~CSCGeometryBuilderFromDDD();

  /// Build the geometry
  void build(boost::shared_ptr<CSCGeometry> geom, const DDCompactView* fv, const MuonDDDConstants& muonConstants);

protected:

private:

  /// Build endcap CSCs
  void buildEndcaps( boost::shared_ptr<CSCGeometry> geom, DDFilteredView* fv, const MuonDDDConstants& muonConstants );

  /// Build one CSC chamber, and its component layers, and add them to the geometry
  void buildChamber (  
	boost::shared_ptr<CSCGeometry> theGeometry,  // the geometry container
	CSCDetId chamberId,              // the DetId of this chamber
        const std::vector<float>& fpar,  // volume parameters
        const std::vector<float>& fupar, // user parameters
        const std::vector<float>& gtran, // translation vector
        const std::vector<float>& grmat, // rotation matrix
        const CSCWireGroupPackage& wg   // wire group info
	);

  const std::string myName;

};
#endif

