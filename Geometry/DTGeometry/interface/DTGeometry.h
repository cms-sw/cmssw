#ifndef DTGeometry_DTGeometry_h
#define DTGeometry_DTGeometry_h

/** \class DTGeometry
 *
 *  The model of the geometry of Muon Drift Tube detectors.
 *
 *  The geometry owns the DTChamber s; these own their DTSuperLayer s which 
 *  in turn own their DTLayer s.
 *
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


class DTGeometry : public TrackingGeometry {

  typedef std::map<DetId, GeomDet*> DTDetMap;

  public:
    /// Default constructor
    DTGeometry();

    /// Destructor
    ~DTGeometry() override;

    void clear();

    //---- Base class' interface 

    // Return a vector of all det types
    const DetTypeContainer&  detTypes() const override;

    // Returm a vector of all GeomDetUnit
    const DetContainer&  detUnits() const override;

    // Returm a vector of all GeomDet (including all GeomDetUnits)
    const DetContainer& dets() const override;

    // Returm a vector of all GeomDetUnit DetIds
    const DetIdContainer&    detUnitIds() const override;

    // Returm a vector of all GeomDet DetIds (including those of GeomDetUnits)
    const DetIdContainer& detIds() const override;

    // Return the pointer to the GeomDetUnit corresponding to a given DetId
    const GeomDet* idToDetUnit(DetId) const override;

    // Return the pointer to the GeomDet corresponding to a given DetId
    const GeomDet* idToDet(DetId) const override;


    //---- Extension of the interface

    /// Return a vector of all Chamber
    const std::vector<const DTChamber*>& chambers() const;

    /// Return a vector of all SuperLayer
    const std::vector<const DTSuperLayer*>& superLayers() const;

    /// Return a vector of all SuperLayer
    const std::vector<const DTLayer*>& layers() const;


    /// Return a DTChamber given its id
    const DTChamber* chamber(const DTChamberId& id) const;

    /// Return a DTSuperLayer given its id
    const DTSuperLayer* superLayer(const DTSuperLayerId& id) const;

    /// Return a layer given its id
    const DTLayer* layer(const DTLayerId& id) const;


  private:
  
    friend class DTGeometryBuilderFromDDD;
    friend class DTGeometryBuilderFromCondDB;

    friend class GeometryAligner;

    void deallocate();

    /// Add a DTChamber to Geometry
    void add(DTChamber* ch);

    /// Add a DTSuperLayer to Geometry
    void add(DTSuperLayer* sl);

    /// Add a DTLayer to Geometry
    void add(DTLayer* l);


    // The chambers are owned by the geometry (and in turn own superlayers
    // and layers)
    std::vector<const DTChamber*> theChambers; 

    // All following pointers are redundant; they are used only for an
    // efficient implementation of the interface, and are NOT owned.

    std::vector<const DTSuperLayer*> theSuperLayers; 
    std::vector<const DTLayer*> theLayers;

    // Map for efficient lookup by DetId 
    DTDetMap          theMap;

    // These are used rarely; they could be computed at runtime 
    // to save memory.
    DetContainer      theDetUnits;       // all layers
    DetContainer      theDets;           // all chambers, SL, layers

    // Replace local static with mutable members
    // to allow lazy evaluation if (ever) needed.
    DetTypeContainer  theDetTypes;
    DetIdContainer    theDetUnitIds;
    DetIdContainer    theDetIds;
};

#endif
