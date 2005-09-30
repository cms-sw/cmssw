#ifndef GEOMETRY_CALOGEOMETRY_CALOSUBDETECTORGEOMETRY_H
#define GEOMETRY_CALOGEOMETRY_CALOSUBDETECTORGEOMETRY_H 1

#include <map>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"

namespace cms {

  /** \class CaloSubdetectorGeometry
      
  $Date: $
  $Revision: $
  \author J. Mans - Minnesota
  */
  class CaloSubdetectorGeometry {
  public:
    virtual ~CaloSubdetectorGeometry() { }

    /// is this detid present in the geometry?
    bool present(const DetId& id) const;
    /// Get the cell geometry of a given detector id.  Should return false if not found.
    const cms::CaloCellGeometry* getGeometry(const DetId& id) const;
    /** \brief Get a list of valid detector ids (for the given subdetector)
	\note The implementation in this class is relevant for SubdetectorGeometries which handle only
	a single subdetector at a time.  It does not look at the det and subdet arguments.
    */
    virtual std::vector<DetId> getValidDetIds(DetId::Detector det, int subdet) const;
    /// Eventually -- get closest cell, etc...
  protected:
    mutable std::vector<DetId> validIds_;
    std::map<DetId, const CaloCellGeometry*> cellGeometries_;    
  };
}

#endif
