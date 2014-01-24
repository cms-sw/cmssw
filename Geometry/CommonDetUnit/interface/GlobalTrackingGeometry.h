#ifndef GlobalTrackingGeometry_h
#define GlobalTrackingGeometry_h

/** \class GlobalTrackingGeometry
 *
 *  Single entry point to the tracker and muon geometries.
 *  The main purpose is to provide the methods idToDetUnit(DetId) and idToDet(DetId)
 *  that allow to get an element of the geometry given its DetId, regardless of wich subdetector it belongs.
 * 
 *  The slave geometries (TrackerGeometry, DTGeometry, CSCGeometry, RPCGeometry, GEMGeometry) are accessible 
 *  with the method slaveGeometry(DetId).
 *
 *  $Date: 2012/07/24 13:48:28 $
 *  $Revision: 1.7 $
 *  \author M. Sani
 */

# include <Geometry/CommonDetUnit/interface/TrackingGeometry.h>
# include <vector>

class GlobalTrackingGeometry : public TrackingGeometry
{
public:
    /// Constructor
    GlobalTrackingGeometry(std::vector<const TrackingGeometry*>& geos);

    /// Destructor
    virtual ~GlobalTrackingGeometry();  

    // Return a vector of all det types.
    virtual const DetTypeContainer&  detTypes()         const;

    // Returm a vector of all GeomDetUnit
    virtual const DetUnitContainer&  detUnits()         const;

    // Returm a vector of all GeomDet (including all GeomDetUnits)
    virtual const DetContainer&      dets()             const;

    // Returm a vector of all GeomDetUnit DetIds
    virtual const DetIdContainer&    detUnitIds()       const;

    // Returm a vector of all GeomDet DetIds (including those of GeomDetUnits)
    virtual const DetIdContainer&    detIds()           const;

    // Return the pointer to the GeomDetUnit corresponding to a given DetId
    virtual const GeomDetUnit*       idToDetUnit(DetId) const;

    // Return the pointer to the GeomDet corresponding to a given DetId
    // (valid also for GeomDetUnits)
    virtual const GeomDet*           idToDet(DetId)     const; 
        
    /// Return the pointer to the actual geometry for a given DetId
    const TrackingGeometry* slaveGeometry(DetId id) const;

private:

    std::vector<const TrackingGeometry*> theGeometries;

    // The const methods claim to simply return these vectors,
    // but actually, they'll fill them up the first time they
    // are called, which is rare (or never).
    mutable DetTypeContainer  theDetTypes;
    mutable DetUnitContainer  theDetUnits; 
    mutable DetContainer      theDets; 
    mutable DetIdContainer    theDetUnitIds;
    mutable DetIdContainer    theDetIds;
};
#endif

