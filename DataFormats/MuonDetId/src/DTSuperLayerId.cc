/** \file
 *  See header file for a description of this class.
 *
 *  $Date: 2005/12/19 16:15:12 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include <iostream>
#include <DataFormats/MuonDetId/interface/DTSuperLayerId.h>
#include <FWCore/Utilities/interface/Exception.h>



DTSuperLayerId::DTSuperLayerId():DTChamberId() {}



DTSuperLayerId::DTSuperLayerId(uint32_t id) : DTChamberId(id) {}// FIXME: Check this is a valid DTSuperLayerId?



DTSuperLayerId::DTSuperLayerId(int wheel,
	       int station,
	       int sector,
	       int superlayer) : DTChamberId(wheel, station, sector) {
		 if(superlayer < minSuperLayerId || superlayer > maxSuperLayerId) {
		   throw cms::Exception("InvalidDetId") << "DTSuperLayerId ctor:" 
							<< " Invalid parameters: " 
							<< " Wh:"<< wheel
							<< " St:"<< station
							<< " Se:"<< sector
							<< " Sl:"<< superlayer
							<< std::endl;
		 }
		 id_ |= (superlayer & slMask_) << slayerStartBit_;
	       }



// Copy Constructor.
DTSuperLayerId::DTSuperLayerId(const DTSuperLayerId& slId) : DTChamberId() {
  id_ = (slId.rawId() &  slIdMask_);						 
}



// Constructor from a camberId and SL number
DTSuperLayerId::DTSuperLayerId(const DTChamberId& chId, int superlayer) : DTChamberId(chId) {
  if(superlayer < minSuperLayerId || superlayer > maxSuperLayerId) {
    throw cms::Exception("InvalidDetId") << "DTSuperLayerId ctor:" 
					 << " Invalid parameter: " 
					 << " Sl:"<< superlayer
					 << std::endl;
  }
  id_ |= (superlayer & slMask_) << slayerStartBit_;
}






std::ostream& operator<<( std::ostream& os, const DTSuperLayerId& id ){
  os << " Wh:"<< id.wheel()
     << " St:"<< id.station() 
     << " Se:"<< id.sector()
     << " Sl:"<< id.superlayer()
     <<" ";

  return os;
}

