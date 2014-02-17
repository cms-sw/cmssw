#ifndef DataFormats_DTRangeMapAccessor_H
#define DataFormats_DTRangeMapAccessor_H

#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include <utility>

/** \class DTSuperLayerIdComparator
 *  Comparator to retrieve by SL objects written into a RangeMap by layer.
 *
 *  $Date: 2006/07/18 08:35:31 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

class DTSuperLayerIdComparator {
public:
  // Operations
  /// Compare two superLayerId
  bool operator()(DTSuperLayerId sl1, DTSuperLayerId sl2) const {
    if (sl1 == sl2) return false;
    return (sl1 < sl2);
  }
  
private:

};



/** \class DTChamberIdComparator
 *  Comparator to retrieve  by chamber objects written into a RangeMap by layer or by SL.
 *
 *  $Date: 2006/07/18 08:35:31 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

class DTChamberIdComparator {
public:
  // Operations
  /// Compare two ChamberId
  bool operator()(DTChamberId ch1, DTChamberId ch2) const {
    if (ch1 == ch2) return false;
    return (ch1<ch2);
  }

private:

};

/** \class DTChamberIdDetLayerComparator
 *  Comparator to retrieve by chamber objects written into a RangeMap by DetLayer.
 *
 *  $Date: 2006/07/18 08:35:31 $
 *  $Revision: 1.2 $
 *  \author M. Sani 
 */

class DTChamberIdDetLayerComparator {
 public:
  bool operator() (DTChamberId ch1, DTChamberId ch2) const {

    if (ch1.station() == ch2.station())
      return false;
    
    return (ch1.station() < ch2.station());
  }
};


/** \class DTRangeMapAccessor
 *  Utility class for access to objects in a RangeMap with needed granularity.
 *
 *  $Date: 2006/07/18 08:35:31 $
 *  $Revision: 1.2 $
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
  static std::pair<DTLayerId, DTSuperLayerIdComparator>
  layersBySuperLayer(DTSuperLayerId slId);

  /// Access by chamber objects written into a RangeMap by layer.
  static std::pair<DTLayerId, DTChamberIdComparator>
  layersByChamber(DTChamberId chamberId);
  
  /// Access by chamber objects written into a RangeMap by SL.
  static std::pair<DTSuperLayerId, DTChamberIdComparator>
  superLayersByChamber(DTChamberId chamberId);
  
  /// Access chambers in a RangeMap by DetLayer.
  static std::pair<DTChamberId, DTChamberIdDetLayerComparator>
    chambersByDetLayer(DTChamberId id);

private:

};

#endif

