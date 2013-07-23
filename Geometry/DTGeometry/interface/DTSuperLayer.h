#ifndef DTSUPERLAYER_H
#define DTSUPERLAYER_H

/** \class DTSuperLayer
 *
 *  Model of a superlayer in Muon Drift Tube chambers.
 *  
 *  A superlayer is composed by 4 staggered DTLayer s.
 *
 *  $date   : 13/01/2006 11:47:03 CET $
 *  $Revision: 1.6 $
 *  \author Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

/* Collaborating Class Declarations */
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

class DTLayer;
class DTChamber;


class DTSuperLayer : public GeomDet {

  public:

/* Constructor */ 
    DTSuperLayer(DTSuperLayerId id,
                 ReferenceCountingPointer<BoundPlane>& plane,
                 const DTChamber* ch=0);

/* Destructor */ 
    virtual ~DTSuperLayer() ;

/* Operations */ 
    /// Return the DetId of this SL
    DTSuperLayerId id() const;

    // Which subdetector
    virtual SubDetector subDetector() const {return GeomDetEnumerators::DT;}

    /// True if id are the same
    bool operator==(const DTSuperLayer& sl) const ;

    /// Return the layers in the SL
    virtual std::vector< const GeomDet*> components() const;

    /// Return the layer with a given id in this SL
    virtual const GeomDet* component(DetId id) const;

    /// Return the layers in the SL
    const std::vector< const DTLayer*>& layers() const;

    /// Add layer to the SL which owns it
    void add(DTLayer* l);

    /// Return the chamber this SL belongs to (0 if any, eg if a SL is
    /// built on his own)
    const DTChamber* chamber() const;

    /// Return the layer corresponding to the given id 
    const DTLayer* layer(DTLayerId id) const;
  
    /// Return the given layer.
    /// Layers are numbered 1-4.
    const DTLayer* layer(int ilay) const;


  private:
    DTSuperLayerId theId;
    // The SL owns its Layer
    std::vector< const DTLayer*> theLayers;
    const DTChamber* theCh;

  protected:

};
#endif // DTSUPERLAYER_H

