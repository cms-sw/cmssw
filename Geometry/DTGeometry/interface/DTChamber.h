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
 *  $Date: 2010/04/09 12:17:26 $
 *  $Revision: 1.7 $
 *  \author S. Lacaprara, N. Amapane
 */

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

class DTSuperLayer;
class DTLayer;

class DTChamber : public GeomDet {

  public:
    /// Constructor
    DTChamber(DTChamberId id, const ReferenceCountingPointer<BoundPlane>& plane);

    /// Destructor
    virtual ~DTChamber();

    /// Return the DTChamberId of this chamber
    DTChamberId id() const;

    // Which subdetector
    virtual SubDetector subDetector() const {return GeomDetEnumerators::DT;}

    /// equal if the id is the same
    bool operator==(const DTChamber& ch) const;

    /// Add SL to the chamber which takes ownership
    void add(DTSuperLayer* sl);

    /// Return the superlayers in the chamber
    virtual std::vector< const GeomDet*> components() const;

    /// Return the sub-component (SL or layer) with a given id in this chamber
    virtual const GeomDet* component(DetId id) const;

    /// Return the superlayers in the chamber
    const std::vector< const DTSuperLayer*>& superLayers() const;

    /// Return the superlayer corresponding to the given id 
    const DTSuperLayer* superLayer(DTSuperLayerId id) const;
  
    /// Return the given superlayer.
    /// Superlayers are numbered 1 (phi), 2 (Z), 3 (phi)
    const DTSuperLayer* superLayer(int isl) const;

    /// Return the layer corresponding to the given id 
    const DTLayer* layer(DTLayerId id) const;

  private:

    DTChamberId theId;

    // The chamber owns its SL
    std::vector<const DTSuperLayer*> theSLs;

};
#endif
