/** \file
 * Impl of DTWireId
 *
 * \author G. Cerminara - INFN Torino
 * \version $Id: DTWireId.cc,v 1.3 2006/04/12 17:52:40 namapane Exp $
 * \date 02 Aug 2005
*/


#include <iostream>
#include <DataFormats/MuonDetId/interface/DTWireId.h>
#include <FWCore/Utilities/interface/Exception.h>



DTWireId::DTWireId() : DTLayerId() {}



DTWireId::DTWireId(uint32_t id) {
  id_= id;
  checkMuonId(); // Check this is a valid id for muon DTs.
}



DTWireId::DTWireId(int wheel,
		   int station,
		   int sector,
		   int superlayer,
		   int layer,
		   int wire) : DTLayerId(wheel, station, sector, superlayer, layer) {
		     if (wire < minWireId || wire > maxWireId) {
		       throw cms::Exception("InvalidDetId") << "DTWireId ctor:" 
							    << " Invalid parameters: " 
							    << " Wh:"<< wheel
							    << " St:"<< station
							    << " Se:"<< sector
							    << " Sl:"<< superlayer
							    << " La:"<< layer
							    << " Wi:"<< wire
							    << std::endl;
		     }
      
		     id_ |= (wire & wireMask_) << wireStartBit_;
		   }



// Copy Constructor.
DTWireId::DTWireId(const DTWireId& wireId) {
  id_ = wireId.rawId();
}



// Constructor from a CamberId and SL, layer and wire numbers
DTWireId::DTWireId(const DTChamberId& chId, int superlayer, int layer, int wire) :
  DTLayerId(chId, superlayer, layer) {
    if (wire < minWireId || wire > maxWireId) {
      throw cms::Exception("InvalidDetId") << "DTWireId ctor:" 
					   << " Invalid parameters: " 
					   << " Wi:"<< wire
					   << std::endl;
    }
      
    id_ |= (wire & wireMask_) << wireStartBit_;
  }



// Constructor from a SuperLayerId and layer and wire numbers
DTWireId::DTWireId(const DTSuperLayerId& slId, int layer, int wire) :
  DTLayerId(slId, layer) {
    if (wire < minWireId || wire > maxWireId) {
      throw cms::Exception("InvalidDetId") << "DTWireId ctor:" 
					   << " Invalid parameters: " 
					   << " Wi:"<< wire
					   << std::endl;
    }
    
    id_ |= (wire & wireMask_) << wireStartBit_;
  }



// Constructor from a layerId and a wire number
DTWireId::DTWireId(const DTLayerId& layerId, int wire) : DTLayerId(layerId) {
  if (wire < minWireId || wire > maxWireId) {
    throw cms::Exception("InvalidDetId") << "DTWireId ctor:" 
					 << " Invalid parameters: " 
					 << " Wi:"<< wire
					 << std::endl;
  }
    
  id_ |= (wire & wireMask_) << wireStartBit_;
}
  


// Ostream operator
std::ostream& operator<<( std::ostream& os, const DTWireId& id ){

  os << " Wh:"<< id.wheel()
     << " St:"<< id.station() 
     << " Se:"<< id.sector()
     << " Sl:"<< id.superlayer()
     << " La:"<< id.layer()
     << " Wi:"<< id.wire()
     <<" ";

  return os;
}


