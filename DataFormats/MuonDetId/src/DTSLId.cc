/** \file
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include <iostream>
#include <DataFormats/MuonDetId/interface/DTSLId.h>
#include <FWCore/Utilities/interface/Exception.h>



DTSLId::DTSLId():DTChamberId() {}



DTSLId::DTSLId(uint32_t id) : DTChamberId(id) {}



DTSLId::DTSLId(int wheel,
	       int station,
	       int sector,
	       int superlayer) : DTChamberId(wheel, station, sector) {
		 if(superlayer < minSuperLayerId || superlayer > maxSuperLayerId) {
		   throw cms::Exception("InvalidDetId") << "DTSLId ctor:" 
							<< " Invalid parameters: " 
							<< " Wh:"<< wheel
							<< " St:"<< station
							<< " Se:"<< sector
							<< " Sl:"<< superlayer
							<< std::endl;
		 }
		 id_ |= (superlayer & slMask_) << slayerStartBit_;
	       }



std::ostream& operator<<( std::ostream& os, const DTSLId& id ){
  os << " Wh:"<< id.wheel()
     << " St:"<< id.station() 
     << " Se:"<< id.sector()
     << " Sl:"<< id.superlayer()
     <<" ";

  return os;
}

