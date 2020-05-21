#ifndef DataFormats_DTRangeMapAccessor_H
#define DataFormats_DTRangeMapAccessor_H

#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include <utility>

/** \class DTSuperLayerIdComparator
 *  Comparator to retrieve by SL objects written into a RangeMap by layer.
 *
 *  \author G. Cerminara - INFN Torino
 */

class DTSuperLayerIdComparator {
public:
  // Operations
  /// Compare two superLayerId
  bool operator()(const DTSuperLayerId& sl1, const DTSuperLayerId& sl2) const {
    if (sl1 == sl2)
      return false;
    return (sl1 < sl2);
  }

private:
};

/** \class DTChamberIdComparator
 *  Comparator to retrieve  by chamber objects written into a RangeMap by layer or by SL.
 *
 *  \author G. Cerminara - INFN Torino
 */

class DTChamberIdComparator {
public:
  // Operations
  /// Compare two ChamberId
  bool operator()(const DTChamberId& ch1, const DTChamberId& ch2) const {
    if (ch1 == ch2)
      return false;
    return (ch1 < ch2);
  }

private:
};

/** \class DTChamberIdDetLayerComparator
 *  Comparator to retrieve by chamber objects written into a RangeMap by DetLayer.
 *
 *  \author M. Sani 
 */

class DTChamberIdDetLayerComparator {
public:
  bool operator()(const DTChamberId& ch1, const DTChamberId& ch2) const {
    if (ch1.station() == ch2.station())
      return false;

    return (ch1.station() < ch2.station());
  }
};

/** \class DTRangeMapAccessor
 *  Utility class for access to objects in a RangeMap with needed granularity.
 *
 *  \author G. Cerminara - INFN Torino
 */

class DTRangeMapAccessor {
public:
  /// Constructor
  DTRangeMapAccessor();

  /// Destructor
  virtual ~DTRangeMapAccessor();

  // Operations

  /// Access by SL objects written into a RangeMap by layer.
  static std::pair<DTLayerId, DTSuperLayerIdComparator> layersBySuperLayer(const DTSuperLayerId& slId);

  /// Access by chamber objects written into a RangeMap by layer.
  static std::pair<DTLayerId, DTChamberIdComparator> layersByChamber(const DTChamberId& chamberId);

  /// Access by chamber objects written into a RangeMap by SL.
  static std::pair<DTSuperLayerId, DTChamberIdComparator> superLayersByChamber(const DTChamberId& chamberId);

  /// Access chambers in a RangeMap by DetLayer.
  static std::pair<DTChamberId, DTChamberIdDetLayerComparator> chambersByDetLayer(const DTChamberId& id);

private:
};

#endif
