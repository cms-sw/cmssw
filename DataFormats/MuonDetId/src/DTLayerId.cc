/** \file
 * Impl of DTLayerId
 *
 * \author Stefano ARGIRO
 * \version $Id: DTLayerId.cc,v 1.3 2006/04/12 17:52:40 namapane Exp $
 * \date 02 Aug 2005
*/

#include <iostream>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>
#include <FWCore/Utilities/interface/Exception.h>



DTLayerId::DTLayerId() : DTSuperLayerId() {}


DTLayerId::DTLayerId(uint32_t id) {
  // Mask the bits outside DTLayerId fields (notably, the wire number)
  id_ = id & layerIdMask_;
  // Check this is a valid id for muon DTs.
  checkMuonId();
}



// Copy Constructor.
DTLayerId::DTLayerId(const DTLayerId& layerId) {
  // The mask is required for proper slicing, i.e. if layerId is
  // actually a derived class.
  id_ = (layerId.rawId() & layerIdMask_);
}



// Constructor from a camberId and SL and layer numbers
DTLayerId::DTLayerId(const DTChamberId& chId, int superlayer, int layer) : DTSuperLayerId(chId, superlayer) {
  if (layer < minLayerId || layer > maxLayerId) {
    throw cms::Exception("InvalidDetId") << "DTLayerId ctor:" 
					 << " Invalid parameters: " 
					 << " La:"<< layer
					 << std::endl;
  }
  id_ |= (layer & lMask_) << layerStartBit_;
}



// Constructor from a SuperLayerId and layer number
DTLayerId::DTLayerId(const DTSuperLayerId& slId, int layer) : DTSuperLayerId(slId) {
  if (layer < minLayerId || layer > maxLayerId) {
    throw cms::Exception("InvalidDetId") << "DTLayerId ctor:" 
					 << " Invalid parameters: " 
					 << " La:"<< layer
					 << std::endl;
  }
  id_ |= (layer & lMask_) << layerStartBit_;
}




DTLayerId::DTLayerId(int wheel,
		     int station,
		     int sector,
		     int superlayer,
		     int layer) : DTSuperLayerId(wheel, station, sector, superlayer) {
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


