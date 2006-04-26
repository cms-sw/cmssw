/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/04/07 15:27:39 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DataFormats/MuonDetId/interface/DTDetIdAccessor.h"

using namespace std;



DTDetIdAccessor::DTDetIdAccessor(){}

DTDetIdAccessor::~DTDetIdAccessor(){}


std::pair<DTSuperLayerId, DTSuperLayerIdComparator>
DTDetIdAccessor::bySuperLayer(const DTSuperLayerId& slId) {
  return make_pair(slId, DTSuperLayerIdComparator());
}


std::pair<DTLayerId, DTChamberIdComparator>
DTDetIdAccessor::byChamber(const DTChamberId& chamberId) {
  return make_pair(DTLayerId(chamberId, 0, 0), DTChamberIdComparator());
}
