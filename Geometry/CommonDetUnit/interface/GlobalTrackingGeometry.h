#ifndef GlobalTrackingGeometry_h
#define GlobalTrackingGeometry_h

/** \class GlobalTrackingGeometry
 *
 *  No description available.
 *
 *  $Date: 2006/05/05 14:22:53 $
 *  $Revision: 1.2 $
 *  \author M. Sani
 */

#include <Geometry/CommonDetUnit/interface/TrackingGeometry.h>
#include <vector>

class GlobalTrackingGeometry : public TrackingGeometry {
    public:
        /// Constructor
        GlobalTrackingGeometry();

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
        
        TrackingGeometry* slaveGeometry(DetId id) const;

    private:
 
        std::vector<TrackingGeometry*> theGeometries;
};
#endif

