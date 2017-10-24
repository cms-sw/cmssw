#ifndef DTGeometryBuilder_DTGeometryBuilderFromDDD_h
#define DTGeometryBuilder_DTGeometryBuilderFromDDD_h

/** \class DTGeometryBuilderFromDDD
 *
 *  Build the DTGeometry from the DDD description.  
 *
 *  \author N. Amapane - CERN. 
 *  \author Port of: MuBarDDDGeomBuilder, MuBarDetBuilder (ORCA) by S. Lacaprara, M. Case
 */

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include <memory>
#include <vector>

class DTGeometry;
class DDCompactView;
class DDFilteredView;
class DTChamber;
class DTSuperLayer;
class DTLayer;
class Bounds;
class MuonDDDConstants;

class DTGeometryBuilderFromDDD {
  public:
    /// Constructor
    DTGeometryBuilderFromDDD();

    /// Destructor
    virtual ~DTGeometryBuilderFromDDD();

    // Operations
    void build(std::shared_ptr<DTGeometry> theGeometry,
               const DDCompactView* cview, 
               const MuonDDDConstants& muonConstants);

  private:
    /// create the chamber
    std::shared_ptr< DTChamber > buildChamber(DDFilteredView& fv, 
					      const std::string& type, 
					      const MuonDDDConstants& muonConstants) const;

    /// create the SL
    std::shared_ptr< DTSuperLayer > buildSuperLayer(DDFilteredView& fv,
						    std::shared_ptr< DTChamber > chamber,
						    const std::string& type, 
						    const MuonDDDConstants& muonConstants) const;

    /// create the layer
    std::shared_ptr< DTLayer > buildLayer(DDFilteredView& fv,
					  std::shared_ptr< DTSuperLayer > sl,
					  const std::string& type, 
					  const MuonDDDConstants& muonConstants) const;

    /// get parameter also for boolean solid.
    std::vector<double> extractParameters(DDFilteredView& fv) const ;

    typedef ReferenceCountingPointer<Plane> RCPPlane;

    RCPPlane plane(const DDFilteredView& fv, 
                   Bounds * bounds) const ;

    void buildGeometry( const std::shared_ptr<DTGeometry>& theGeometry,
			DDFilteredView& fv,
			const MuonDDDConstants& muonConstants) const;

};
#endif

