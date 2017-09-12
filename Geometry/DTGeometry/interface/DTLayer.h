#ifndef DTLAYER_H
#define DTLAYER_H

/** \class DTLayer
 *
 *  Model of a layer (row of cells) in Muon Drift Tube chambers.
 *
 *  The layer is the GeomDetUnit for the DTs. 
 *  The individual channes are modelled by DTTopology.
 *
 *  \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

/* Collaborating Class Declarations */
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"
#include "Geometry/DTGeometry/interface/DTLayerType.h"
class DTSuperLayer;
class DTChamber;


class DTLayer : public GeomDetUnit {

  public:

/* Constructor */ 
    DTLayer(const DTLayerId& id,
            ReferenceCountingPointer<BoundPlane>& plane,
            const DTTopology& topo,
            const DTLayerType& type,
            const DTSuperLayer* sl=nullptr) ;

/* Destructor */ 
    ~DTLayer() override ;

/* Operations */ 
    const Topology& topology() const override;

    const GeomDetType& type() const override;

    const DTTopology& specificTopology() const;

    /// Return the DetId of this SL
    DTLayerId id() const;

    /// Return the Superlayer this Layer belongs to (0 if any, eg if a
    /// layer is built on his own)
    const DTSuperLayer* superLayer() const ;

    /// Return the chamber this Layer belongs to (0 if none, eg if a layer is
    /// built on his own)
    const DTChamber* chamber() const;

    /// True if the id are the same
    bool operator==(const DTLayer& l) const;

    /// A Layer has no components
    std::vector< const GeomDet*> components() const override;

  private:
    DTLayerId   theId;
    DTTopology  theTopo;
    DTLayerType theType;

    const DTSuperLayer*   theSL;
  protected:

};
#endif // DTLAYER_H

