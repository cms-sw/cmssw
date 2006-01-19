/** \file
 *  See header file for a description of this class.
 *
 *  $Date: 2005/12/19 16:15:12 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include <iostream>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <FWCore/Utilities/interface/Exception.h>

using namespace std;

DTChamberId::DTChamberId():DetId(DetId::Muon, MuonSubdetId::DT){}


// FIXME: Check this is a valid DTChamberId?
DTChamberId::DTChamberId(uint32_t id):DetId(id) {
  checkMuonId();
//   cout << "DTChamberId::DTChamberId(uint32_t id):DetId(id)" << endl;
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



DTChamberId::DTChamberId(const DTChamberId& chId):
  DetId(chId.rawId() & chamberIdMask_) {
//     cout << "DTChamberId::DTChamberId(const DTChamberId& chId)" << endl;
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



