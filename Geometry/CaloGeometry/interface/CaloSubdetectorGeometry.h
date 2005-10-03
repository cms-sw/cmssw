#ifndef GEOMETRY_CALOGEOMETRY_CALOSUBDETECTORGEOMETRY_H
#define GEOMETRY_CALOGEOMETRY_CALOSUBDETECTORGEOMETRY_H 1

#include <map>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"

class CaloCellGeometry;

/** \class CaloSubdetectorGeometry
      
$Date: 2005/09/30 13:42:44 $
$Revision: 1.1 $
\author J. Mans - Minnesota
*/
class CaloSubdetectorGeometry {
public:
  /// The base class does not assume that it owns the CaloCellGeometry objects
  virtual ~CaloSubdetectorGeometry() { }

  /// Add a cell to the geometry
  void addCell(const DetId& id, const CaloCellGeometry* ccg);
  /// is this detid present in the geometry?
  bool present(const DetId& id) const;
  /// Get the cell geometry of a given detector id.  Should return false if not found.
  const CaloCellGeometry* getGeometry(const DetId& id) const;
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


#endif
