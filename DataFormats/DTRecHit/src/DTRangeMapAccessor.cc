/** \file
 *
 *  $Date: 2006/07/18 08:35:42 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DataFormats/DTRecHit/interface/DTRangeMapAccessor.h"

using namespace std;



DTRangeMapAccessor::DTRangeMapAccessor(){}



DTRangeMapAccessor::~DTRangeMapAccessor(){}



// Access by SL objects written into a RangeMap by layer.
pair<DTLayerId, DTSuperLayerIdComparator>
DTRangeMapAccessor::layersBySuperLayer(DTSuperLayerId slId) {
  return make_pair(DTLayerId(slId,0), DTSuperLayerIdComparator());
}



// Access by chamber objects written into a RangeMap by layer.
pair<DTLayerId, DTChamberIdComparator>
DTRangeMapAccessor::layersByChamber(DTChamberId chamberId) {
  return make_pair(DTLayerId(chamberId,0,0), DTChamberIdComparator());
}



// Access by chamber objects written into a RangeMap by SL.
pair<DTSuperLayerId, DTChamberIdComparator>
DTRangeMapAccessor::superLayersByChamber(DTChamberId chamberId) {
  return make_pair(DTSuperLayerId(chamberId,0), DTChamberIdComparator());
}


// Access by chamber objects written into a RangeMap by DetLayer.
pair<DTChamberId, DTChamberIdDetLayerComparator>
DTRangeMapAccessor::chambersByDetLayer(DTChamberId chamberId) {
  return make_pair(DTChamberId(chamberId), DTChamberIdDetLayerComparator());
}

