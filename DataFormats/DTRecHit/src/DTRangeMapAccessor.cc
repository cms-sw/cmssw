/** \file
 *
 *  \author G. Cerminara - INFN Torino
 */

#include "DataFormats/DTRecHit/interface/DTRangeMapAccessor.h"

using namespace std;

DTRangeMapAccessor::DTRangeMapAccessor() {}

DTRangeMapAccessor::~DTRangeMapAccessor() {}

// Access by SL objects written into a RangeMap by layer.
pair<DTLayerId, DTSuperLayerIdComparator> DTRangeMapAccessor::layersBySuperLayer(const DTSuperLayerId& slId) {
  return make_pair(DTLayerId(slId, 0), DTSuperLayerIdComparator());
}

// Access by chamber objects written into a RangeMap by layer.
pair<DTLayerId, DTChamberIdComparator> DTRangeMapAccessor::layersByChamber(const DTChamberId& chamberId) {
  return make_pair(DTLayerId(chamberId, 0, 0), DTChamberIdComparator());
}

// Access by chamber objects written into a RangeMap by SL.
pair<DTSuperLayerId, DTChamberIdComparator> DTRangeMapAccessor::superLayersByChamber(const DTChamberId& chamberId) {
  return make_pair(DTSuperLayerId(chamberId, 0), DTChamberIdComparator());
}

// Access by chamber objects written into a RangeMap by DetLayer.
pair<DTChamberId, DTChamberIdDetLayerComparator> DTRangeMapAccessor::chambersByDetLayer(const DTChamberId& chamberId) {
  return make_pair(DTChamberId(chamberId), DTChamberIdDetLayerComparator());
}
