#ifndef DTGeometry_DTGeometry_h
#define DTGeometry_DTGeometry_h

/** \class DTGeometry
 *
 *  The model of the geometry of Muon Drift Tube detectors.
 *
 *  The geometry owns the DTChamber s; these own their DTSuperLayer s which 
 *  in turn own their DTLayer s.
 *
 *  $Date: 2006/02/22 11:06:51 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include <DataFormats/DetId/interface/DetId.h>
#include <Geometry/CommonDetUnit/interface/TrackingGeometry.h>
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include <vector>
#include <map>

class GeomDetType;
class GeomDetUnit;

class DTGeometry : public TrackingGeometry {

  typedef std::map<DetId, DTChamber*>     DTChamberMap ;
  typedef std::map<DetId, DTSuperLayer*>  DTSuperLayerMap ;
  typedef std::map<DetId, DTLayer*>       DTLayerMap ;

  public:
    /// Default constructor
    DTGeometry();

    /// Destructor
    virtual ~DTGeometry();

    //---- Base class' interface 

    // Return a vector of all det types
    virtual const DetTypeContainer&  detTypes() const;

    // Returm a vector of all GeomDetUnit
    virtual const DetUnitContainer&  detUnits() const;

    // Returm a vector of all GeomDet (including all GeomDetUnits)
    virtual const DetContainer& dets() const;

    // Returm a vector of all GeomDetUnit DetIds
    virtual const DetIdContainer&    detUnitIds() const;

    // Returm a vector of all GeomDet DetIds (including those of GeomDetUnits)
    virtual const DetIdContainer& detIds() const;

    // Return the pointer to the GeomDetUnit corresponding to a given DetId
    virtual const GeomDetUnit* idToDetUnit(DetId) const;

    // Return the pointer to the GeomDetUnit corresponding to a given DetId
    virtual const GeomDet* idToDet(DetId) const;


    //---- Extension of the interface

    /// Return a vector of all Chamber
    const std::vector<DTChamber*>& chambers() const;

    /// Return a vector of all SuperLayer
    const std::vector<DTSuperLayer*>& superLayers() const;

    /// Return a vector of all SuperLayer
    const std::vector<DTLayer*>& layers() const;


    /// Return a DTChamber given its id
    const DTChamber* chamber(const DTChamberId& id) const;

    /// Return a DTSuperLayer given its id
    const DTSuperLayer* superLayer(const DTSuperLayerId& id) const;

    /// Return a layer given its id
    const DTLayer* layer(const DTLayerId& id) const;

    /// Add a DTChamber to Geometry
    void add(DTChamber* ch);

    /// Add a DTSuperLayer to Geometry
    void add(DTSuperLayer* ch);

    /// Add a DTLayer to Geometry
    void add(DTLayer* ch);

  private:
    // The chambers are owned by the geometry (and in turn own superlayers
    // and layers)
    std::vector<DTChamber*> theChambers; 

    // All following pointers are redundant; they are used only for an
    // efficient implementation of the interface, and are NOT owned.
    std::vector<DTSuperLayer*> theSuperLayers; 
    std::vector<DTLayer*> theLayers;
    DetTypeContainer  theDetTypes;       // the DetTypes
    DetUnitContainer  theDetUnits;       // all layers
    DetContainer      theDets;           // all chambers, SL, layers
    DetIdContainer    theDetUnitIds;     // DetIds of layers
    DetIdContainer    theDetIds;         // DetIds of all chambers, SL, layers
    DTChamberMap      theChambersMap;    // chamber lookup by DetId
    DTSuperLayerMap   theSuperLayersMap; // SL lookup by DetId
    DTLayerMap        theLayersMap;      // layer lookup by DetId 

};

#endif
