#ifndef DTGeometry_DTChamber_h
#define DTGeometry_DTChamber_h

/** \class DTChamber
 *
 *  Model of a Muon Drift Tube chamber.
 *   
 *  A chamber is a GeomDet; the associated reconstructed objects are 
 *  4D segments built in the chamber local reference frame.
 *  The chamber is composed by 2 or three DTSuperLayer, which in turn are 
 *  composed by four DTLayer each.
 *
 *  $Date: 2006/02/02 18:05:54 $
 *  $Revision: 1.3 $
 *  \author S. Lacaprara, N. Amapane
 */

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

class DTSuperLayer;

class DTChamber : public GeomDet {

  public:
    /// Constructor
    DTChamber(DTChamberId id, const ReferenceCountingPointer<BoundPlane>& plane);

    /// Destructor
    virtual ~DTChamber();

    /// Return the DetId of this chamber
    virtual DetId geographicalId() const;

    /// Return the DTChamberId of this chamber
    DTChamberId id() const;

    /// equal if the id is the same
    bool operator==(const DTChamber& ch) const;

    /// Add SL to the chamber which takes ownership
    void add(DTSuperLayer* sl);

    /// Return the superlayers in the chamber
    virtual std::vector< const GeomDet*> components() const;

    /// Return the superlayers in the chamber
    std::vector< const DTSuperLayer*> superLayers() const;

  private:

    DTChamberId theId;

    // The chamber owns its SL
    std::vector<const DTSuperLayer*> theSLs;

};
#endif
