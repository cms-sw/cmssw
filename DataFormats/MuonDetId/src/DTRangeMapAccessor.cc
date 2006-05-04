/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/04/26 07:49:56 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DataFormats/MuonDetId/interface/DTRangeMapAccessor.h"

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


