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
class CSCGeometry;
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

  const std::string myName;

};
#endif

