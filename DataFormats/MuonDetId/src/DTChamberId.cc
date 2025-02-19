/** \file
 *  See header file for a description of this class.
 *
 *  $Date: 2008/11/06 10:34:56 $
 *  $Revision: 1.7 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h" 
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>

using namespace std;

DTChamberId::DTChamberId() : DetId(DetId::Muon, MuonSubdetId::DT){}


DTChamberId::DTChamberId(uint32_t id) :
  DetId(id & chamberIdMask_) { // Mask the bits outside DTChamberId fields
  checkMuonId();               // Check this is a valid id for muon DTs.
}
DTChamberId::DTChamberId(DetId id) :
  DetId(id.rawId() & chamberIdMask_) { // Mask the bits outside DTChamberId fields
  checkMuonId();               // Check this is a valid id for muon DTs.
}



DTChamberId::DTChamberId(int wheel, int station, int sector):
  DetId(DetId::Muon, MuonSubdetId::DT) {
    // Check that arguments are within the range
    if (wheel      < minWheelId      || wheel      > maxWheelId ||
	station    < minStationId    || station    > maxStationId ||
	sector     < minSectorId     || sector     > maxSectorId) {
      throw cms::Exception("InvalidDetId") << "DTChamberId ctor:" 
					   << " Invalid parameters: " 
					   << " Wh:"<< wheel
					   << " St:"<< station
					   << " Se:"<< sector
					   << std::endl;
    }

    int tmpwheelid = wheel- minWheelId +1;
    id_ |= (tmpwheelid& wheelMask_) << wheelStartBit_   |
      (station & stationMask_)      << stationStartBit_ |
      (sector  &sectorMask_ )       << sectorStartBit_;

}



DTChamberId::DTChamberId(const DTChamberId& chId) :
  DetId(chId.rawId() & chamberIdMask_) {   // The mask is required for proper slicing, i.e. if chId is actually a derived class.
}



void DTChamberId::checkMuonId() {
  if (det()!=DetId::Muon || subdetId()!=MuonSubdetId::DT) {
    throw cms::Exception("InvalidDetId") << "DTChamberId ctor:"
					 << " det: " << det()
					 << " subdet: " << subdetId()
					 << " is not a valid DT id";  
  }
}



std::ostream& operator<<( std::ostream& os, const DTChamberId& id ){
  os << " Wh:"<< id.wheel()
     << " St:"<< id.station() 
     << " Se:"<< id.sector()
     <<" ";

  return os;
}



