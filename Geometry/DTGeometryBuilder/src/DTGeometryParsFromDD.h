#ifndef DTGeometryBuilder_DTGeometryParsFromDD_h
#define DTGeometryBuilder_DTGeometryParsFromDD_h

/** \class DTGeometryParsFromDD
 *
 *  Build the DTGeometry from the DDD description.  
 *
 *  $Date: 2009/01/16 11:11:47 $
 *  $Revision: 1.1 $
 *  \author Stefano Lacaprara  <lacaprara@pd.infn.it>  INFN LNL
 */

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
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
class RecoIdealGeometry;

class DTGeometryParsFromDD {
  public:
    /// Constructor
    DTGeometryParsFromDD();

    /// Destructor
    virtual ~DTGeometryParsFromDD();

    // Operations
    void build(const DDCompactView* cview, 
               const MuonDDDConstants& muonConstants,
               RecoIdealGeometry& rig) ;

    enum DTDetTag { DTChamberTag, DTSuperLayerTag, DTLayerTag };
  private:
    /// create the chamber
    void insertChamber(DDFilteredView& fv, 
                       const std::string& type, 
                       const MuonDDDConstants& muonConstants,
                       RecoIdealGeometry& rig) const;

    /// create the SL
    void insertSuperLayer(DDFilteredView& fv,
                          const std::string& type, 
                          const MuonDDDConstants& muonConstants,
                          RecoIdealGeometry& rig) const;

    /// create the layer
    void insertLayer(DDFilteredView& fv,
                     const std::string& type, 
                     const MuonDDDConstants& muonConstants,
                     RecoIdealGeometry& rig) const;

    /// get parameter also for boolean solid.
    std::vector<double> extractParameters(DDFilteredView& fv) const ;

    typedef std::pair<std::vector<double>, std::vector<double> > PosRotPair;

    PosRotPair plane(const DDFilteredView& fv) const ;

    void buildGeometry(DDFilteredView& fv,
                       const MuonDDDConstants& muonConstants,
                       RecoIdealGeometry& rig) const;

};
#endif

