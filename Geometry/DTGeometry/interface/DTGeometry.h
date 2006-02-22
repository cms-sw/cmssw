#ifndef DTGeometry_DTGeometry_h
#define DTGeometry_DTGeometry_h

/** \class DTGeometry
 *
 *  The model of the geometry of Muon Drift Tube detectors.
 *
 *  The geometry owns the DTChamber s; these own their DTSuperLayer s which 
 *  in turn own their DTLayer s.
 *
 *  $Date: 2006/02/07 18:06:36 $
 *  $Revision: 1.6 $
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

    /// Return a vector of all det types
    virtual const DetTypeContainer&  detTypes() const;

    /// Return a vector of all GeomDetUnit
    virtual const DetContainer& dets() const;

    /// Return a vector of all Chamber
    const std::vector<DTChamber*>& chambers() const;

    /// Return a vector of all SuperLayer
    const std::vector<DTSuperLayer*>& superLayers() const;

    /// Return a vector of all SuperLayer
    const std::vector<DTLayer*>& layers() const;

    /// Return a vector of all DetIds
    virtual const DetIdContainer& detIds() const;

    /// Return the pointer to the GeomDetUnit corresponding to a given DetId
    virtual const GeomDetUnit* idToDet(DetId) const;

    /// Return a DTChamber given its id
    const DTChamber* chamber(const DTChamberId& id) const;

    /// Return a DTSuperLayer given its id
    DTSuperLayer* superLayer(const DTSuperLayerId& id) const;

    /// Return a layer given its id
    DTLayer* layer(const DTLayerId& id) const;

    /// Add a DTChamber to Geometry
    void add(DTChamber* ch);

    /// Add a DTSuperLayer to Geometry
    void add(DTSuperLayer* ch);

    /// Add a DTLayer to Geometry
    void add(DTLayer* ch);

  private:
    DetTypeContainer  theDetTypes;
    DetContainer      theDets;
    DTChamberMap      theChambersMap;
    std::vector<DTChamber*> theChambers;
    DTSuperLayerMap   theSuperLayersMap;
    std::vector<DTSuperLayer*> theSuperLayers;
    DTLayerMap        theLayersMap;
    std::vector<DTLayer*> theLayers;
    DetIdContainer    theDetIds;
    mapIdToDet        theMap;

};

#endif
