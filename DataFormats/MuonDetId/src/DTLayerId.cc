/** \file
 * Impl of DTLayerId
 *
 * \author Stefano ARGIRO
 * \version $Id: DTLayerId.cc,v 1.4 2005/10/24 15:56:19 namapane Exp $
 * \date 02 Aug 2005
*/

#include <iostream>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>
#include <FWCore/Utilities/interface/Exception.h>




DTLayerId::DTLayerId() : DTSLId() {}



DTLayerId::DTLayerId(uint32_t id) : DTSLId(id) {}



DTLayerId::DTLayerId(int wheel,
		     int station,
		     int sector,
		     int superlayer,
		     int layer) : DTSLId(wheel, station, sector, superlayer) {
		       if (layer < minLayerId || layer > maxLayerId) {
			 throw cms::Exception("InvalidDetId") << "DTLayerId ctor:" 
							      << " Invalid parameters: " 
							      << " Wh:"<< wheel
							      << " St:"<< station
							      << " Se:"<< sector
							      << " Sl:"<< superlayer
							      << " La:"<< layer
							      << std::endl;
		       }
		       
		       id_ |= (layer & lMask_) << layerStartBit_;
		     }



std::ostream& operator<<( std::ostream& os, const DTLayerId& id ){
  os << " Wh:"<< id.wheel()
     << " St:"<< id.station() 
     << " Se:"<< id.sector()
     << " Sl:"<< id.superlayer()
     << " La:"<< id.layer()
     <<" ";

  return os;
}


