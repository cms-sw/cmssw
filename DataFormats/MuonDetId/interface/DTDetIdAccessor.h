#ifndef DataFormats_DTDetIdAccessor_H
#define DataFormats_DTDetIdAccessor_H

#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include <utility>

/** \class DTSuperLayerIdComparator
 *  Comparator to retrieve objects from RangeMap by SL.
 *
 *  $Date: 2006/04/07 15:27:38 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

class DTSuperLayerIdComparator {
public:
  // Operations
  /// Compare two layerId
  bool operator()(const DTSuperLayerId l1, const DTSuperLayerId l2) const {
    if (l1 == l2) return false;
    return (l1 < l2);
  }

protected:

private:

};

/** \class DTChamberIdComparator
 *  Comparator to retrieve objects from RangeMap by Chamber.
 *
 *  $Date: 2006/04/07 15:27:38 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

class DTChamberIdComparator {
public:
  // Operations
  /// Compare two layerId
  bool operator()(const DTLayerId& l1, const DTLayerId& l2) const {
    if (l1.chamberId() == l2.chamberId()) return false;
    return (l1.chamberId()<l2.chamberId());
  }

protected:

private:

};


/** \class DTDetIdAccessor
 *  Utility class for access to objects in a RangeMap with needed granularity.
 *
 *  $Date: 2006/04/07 15:27:38 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

class DTDetIdAccessor {
public:
  /// Constructor
  DTDetIdAccessor();

  /// Destructor
  virtual ~DTDetIdAccessor();

  // Operations

  /// Access by superlayer
  static std::pair<DTSuperLayerId, DTSuperLayerIdComparator> bySuperLayer(const DTSuperLayerId& slId);

  /// Access by chamber
  static std::pair<DTLayerId, DTChamberIdComparator> byChamber(const DTChamberId& chamberId);
  


protected:

private:

};





#endif

