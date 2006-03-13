#ifndef CSCGeometryBuilder_CSCGeometryBuilderFromDDD_h
#define CSCGeometryBuilder_CSCGeometryBuilderFromDDD_h

/** \class CSCGeometryBuilderFromDDD
 *
 *  Build the CSCGeometry from the DDD description.
 *
 *  \author Tim Cox
 */

#include <string>

class DDCompactView;
class DDFilteredView;
class CSCGeometry;
class CSCWireGroupPackage;

class CSCGeometryBuilderFromDDD {
public:
  /// Constructor
  CSCGeometryBuilderFromDDD();

  /// Destructor
  virtual ~CSCGeometryBuilderFromDDD();

  /// Build the geometry
  CSCGeometry* build(const DDCompactView* fv);

protected:

private:

  /// Build endcap CSCs
  CSCGeometry* buildEndcaps( DDFilteredView* fv );

  /// Build one CSC layer and add it to the geometry
  void buildLayer (  
	CSCGeometry* theGeometry,        // the geometry container
	int    detid,                    // packed index from CSCDetId
        const std::vector<float>& fpar,  // volume parameters
        const std::vector<float>& fupar, // user parameters
        const std::vector<float>& gtran, // translation vector
        const std::vector<float>& grmat, // rotation matrix
        const CSCWireGroupPackage& wg    // wire group info
	);

  static const std::string myName;

  // For want of anything better, output is flagged by this thing
  // which is set at compile. Dark ages stuff.
  static bool debugV;

};
#endif

