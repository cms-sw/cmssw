#ifndef DTGeometryBuilder_DTGeometryBuilderFromDDD_h
#define DTGeometryBuilder_DTGeometryBuilderFromDDD_h

/** \class DTGeometryBuilderFromDDD
 *
 *  Build the DTGeometry from the DDD description.  
 *
 *  $Date: 2006/02/01 16:38:40 $
 *  $Revision: 1.4 $
 *  \author N. Amapane - CERN. 
 *  \author Port of: MuBarDDDGeomBuilder, MuBarDetBuilder (ORCA) by S. Lacaprara, M. Case
 */

#include "Geometry/Surface/interface/BoundPlane.h"
#include <vector>

class DTGeometry;
class DDCompactView;
class DDFilteredView;
class DTChamber;
class DTSuperLayer;
class DTLayer;
class Bounds;

class DTGeometryBuilderFromDDD {
  public:
    /// Constructor
    DTGeometryBuilderFromDDD();

    /// Destructor
    virtual ~DTGeometryBuilderFromDDD();

    // Operations
    DTGeometry* build(const DDCompactView* cview);

  private:
    /// create the chamber
    DTChamber* buildChamber(DDFilteredView& fv, 
                            const std::string& type) const;

    /// create the SL
    DTSuperLayer* buildSuperLayer(DDFilteredView& fv,
                                  DTChamber* chamber,
                                  const std::string& type) const;

    /// create the layer
    DTLayer* buildLayer(DDFilteredView& fv,
                        DTSuperLayer* sl,
                        const std::string& type) const;

    /// get parameter also for boolean solid.
    std::vector<double> extractParameters(DDFilteredView& fv) const ;

    typedef ReferenceCountingPointer<BoundPlane> RCPPlane;

    RCPPlane plane(const DDFilteredView& fv, 
                   const Bounds& bounds) const ;

    DTGeometry* buildGeometry(DDFilteredView& fv) const;

};
#endif

