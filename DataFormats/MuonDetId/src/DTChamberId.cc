/** \file
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include <iostream>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <FWCore/Utilities/interface/Exception.h>


DTChamberId::DTChamberId():DetId(DetId::Muon, MuonSubdetId::DT){}



DTChamberId::DTChamberId(uint32_t id):DetId(id) {
  if (det()!=DetId::Muon || subdetId()!=MuonSubdetId::DT) {
    throw cms::Exception("InvalidDetId") << "DTChamberId ctor:"
					 << " det: " << det()
					 << " subdet: " << subdetId()
					 << " is not a valid DT id";  
  }
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



std::ostream& operator<<( std::ostream& os, const DTChamberId& id ){
  os << " Wh:"<< id.wheel()
     << " St:"<< id.station() 
     << " Se:"<< id.sector()
     <<" ";

  return os;
}



