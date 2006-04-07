/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "DataFormats/MuonDetId/interface/DTDetIdAccessor.h"

using namespace std;



DTDetIdAccessor::DTDetIdAccessor(){}

DTDetIdAccessor::~DTDetIdAccessor(){}


std::pair<DTLayerId, DTSuperLayerIdComparator>
DTDetIdAccessor::bySuperLayer(const DTSuperLayerId& slId) {
  return make_pair(DTLayerId(slId, 0), DTSuperLayerIdComparator());
}


std::pair<DTLayerId, DTChamberIdComparator>
DTDetIdAccessor::byChamber(const DTChamberId& chamberId) {
  return make_pair(DTLayerId(chamberId, 0, 0), DTChamberIdComparator());
}
