#ifndef DTGeometryBuilder_DTGeometryBuilderFromDDD_h
#define DTGeometryBuilder_DTGeometryBuilderFromDDD_h

/** \class DTGeometryBuilderFromDDD
 *
 *  Build the DTGeometry from the DDD description.  
 *
 *  $Date: 2012/12/24 14:35:11 $
 *  $Revision: 1.5 $
 *  \author N. Amapane - CERN. 
 *  \author Port of: MuBarDDDGeomBuilder, MuBarDetBuilder (ORCA) by S. Lacaprara, M. Case
 */

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include <vector>
#include <boost/shared_ptr.hpp>

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
    void build(boost::shared_ptr<DTGeometry> theGeometry,
               const DDCompactView* cview, 
               const MuonDDDConstants& muonConstants);

  private:
    /// create the chamber
    DTChamber* buildChamber(DDFilteredView& fv, 
                            const std::string& type, 
                            const MuonDDDConstants& muonConstants) const;

    /// create the SL
    DTSuperLayer* buildSuperLayer(DDFilteredView& fv,
                                  DTChamber* chamber,
                                  const std::string& type, 
                                  const MuonDDDConstants& muonConstants) const;

    /// create the layer
    DTLayer* buildLayer(DDFilteredView& fv,
                        DTSuperLayer* sl,
                        const std::string& type, 
                        const MuonDDDConstants& muonConstants) const;

    /// get parameter also for boolean solid.
    std::vector<double> extractParameters(DDFilteredView& fv) const ;

    typedef ReferenceCountingPointer<Plane> RCPPlane;

    RCPPlane plane(const DDFilteredView& fv, 
                   Bounds * bounds) const ;

    void buildGeometry(boost::shared_ptr<DTGeometry> theGeometry,
                       DDFilteredView& fv,
                       const MuonDDDConstants& muonConstants) const;

};
#endif

