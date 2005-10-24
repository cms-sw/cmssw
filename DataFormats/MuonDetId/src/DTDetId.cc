/** \file
 * Impl of DTDetId
 *
 * \author Stefano ARGIRO
 * \version $Id: DTDetId.cc,v 1.3 2005/10/18 17:57:47 namapane Exp $
 * \date 02 Aug 2005
*/

#include <iostream>
#include <DataFormats/MuonDetId/interface/DTDetId.h>
#include <FWCore/Utilities/interface/Exception.h>

DTDetId::DTDetId():DetId(DetId::Muon, MuonSubdetId::DT){}

DTDetId::DTDetId(uint32_t id):DetId(id) {
  if (det()!=DetId::Muon || subdetId()!=MuonSubdetId::DT) {
    throw cms::Exception("InvalidDetId") << "DTDetId ctor:"
					 << " det: " << det()
					 << " subdet: " << subdetId()
					 << " is not a valid DT id";  
  }
}

DTDetId::DTDetId(int wheel, int station, int sector, int superlayer,
		 int layer, int wire) : 
  DetId(DetId::Muon, MuonSubdetId::DT) 
{
  if (wheel      < minWheelId      || wheel      > maxWheelId ||
      station    < minStationId    || station    > maxStationId ||
      sector     < minSectorId     || sector     > maxSectorId ||
      superlayer < minSuperLayerId || superlayer > maxSuperLayerId ||
      layer      < minLayerId      || layer      > maxLayerId ||
      wire       < minWireId       || wire       > maxWireId) {
    throw cms::Exception("InvalidDetId") << "DTDetId ctor:" 
					 << " Invalid parameters: " 
					 << " Wh:"<< wheel
					 << " St:"<< station
					 << " Se:"<< sector
					 << " Sl:"<< superlayer
					 << " La:"<< layer
					 << " Wi:"<< wire
					 << std::endl;
  }
      
  int tmpwheelid = wheel- minWheelId +1;
  id_ |= (tmpwheelid& wheelMask_) << wheelStartBit_   |
    (station & stationMask_)      << stationStartBit_ |
    (sector  &sectorMask_ )       << sectorStartBit_  |
    (superlayer & slMask_)        << slayerStartBit_  |
    (layer & lMask_)              << layerStartBit_   |
    (wire & wireMask_)            << wireStartBit_;
}



std::ostream& operator<<( std::ostream& os, const DTDetId& id ){

  os << " Wh:"<< id.wheel()
     << " St:"<< id.station() 
     << " Se:"<< id.sector()
     << " Sl:"<< id.superlayer()
     << " La:"<< id.layer()
     << " Wi:"<< id.wire()
     <<" ";

  return os;
}


