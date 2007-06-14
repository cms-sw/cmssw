#ifndef GEOMETRY_CALOGEOMETRY_CALOSUBDETECTORGEOMETRY_H
#define GEOMETRY_CALOGEOMETRY_CALOSUBDETECTORGEOMETRY_H 1

#include <ext/hash_map>
#include <vector>
#include <set>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class CaloCellGeometry;

/** \class CaloSubdetectorGeometry
      
Base class for a geometry container for a specific calorimetry
subdetector.


$Date: 2007/03/07 09:18:00 $
$Revision: 1.7 $
\author J. Mans - Minnesota
*/
class CaloSubdetectorGeometry {
public:
  typedef  __gnu_cxx::hash_map< unsigned int, CaloCellGeometry const *> CellCont;

  /// The base class does not assume that it owns the CaloCellGeometry objects
  virtual ~CaloSubdetectorGeometry();

  /// the cells
  CellCont const & cellGeometries() const { return cellGeometries_; }  

  /// Add a cell to the geometry
  void addCell(const DetId& id, const CaloCellGeometry* ccg);

  /// is this detid present in the geometry?
  virtual bool present(const DetId& id) const;

  /// Get the cell geometry of a given detector id.  Should return false if not found.
  virtual const CaloCellGeometry* getGeometry(const DetId& id) const;

  /** \brief Get a list of valid detector ids (for the given subdetector)
      \note The implementation in this class is relevant for SubdetectorGeometries which handle only
      a single subdetector at a time.  It does not look at the det and subdet arguments.
  */
  virtual std::vector<DetId> const & getValidDetIds(DetId::Detector det, int subdet) const;

  // Get closest cell, etc...
  virtual DetId getClosestCell(const GlobalPoint& r) const ;

  typedef std::set<DetId> DetIdSet;

  /** \brief Get a list of all cells within a dR of the given cell

      The default implementation makes a loop over all cell geometries.
      Cleverer implementations are suggested to use rough conversions between
      eta/phi and ieta/iphi and test on the boundaries.
  */
  virtual DetIdSet getCells(const GlobalPoint& r, double dR) const;

  static double deltaR(const GlobalPoint& p1, const GlobalPoint& p2);

  //FIXME: Hcal implements its own  getValidDetId....
protected:
  mutable std::vector<DetId> validIds_;

private:
  CellCont cellGeometries_;    
};


#endif
