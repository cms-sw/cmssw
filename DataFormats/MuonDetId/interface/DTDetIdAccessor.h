#ifndef DataFormats_DTDetIdAccessor_H
#define DataFormats_DTDetIdAccessor_H

#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include <utility>

/** \class DTSuperLayerIdComparator
 *  Comparator to retrieve objects from RangeMap by SL.
 *
 *  $Date: 2006/03/20 12:42:28 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - INFN Torino
 */

class DTSuperLayerIdComparator {
public:
  // Operations
  /// Compare two layerId
  bool operator()(const DTLayerId& l1, const DTLayerId& l2) const {
    if (l1.superlayerId() == l2.superlayerId()) return false;
    return (l1.superlayerId()<l2.superlayerId());
  }

protected:

private:

};

/** \class DTChamberIdComparator
 *  Comparator to retrieve objects from RangeMap by Chamber.
 *
 *  $Date: 2006/03/20 12:42:28 $
 *  $Revision: 1.3 $
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
 *  $Date: $
 *  $Revision: $
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
  static std::pair<DTLayerId, DTSuperLayerIdComparator> bySuperLayer(const DTSuperLayerId& slId);

  /// Access by chamber
  static std::pair<DTLayerId, DTChamberIdComparator> byChamber(const DTChamberId& chamberId);
  


protected:

private:

};





#endif

