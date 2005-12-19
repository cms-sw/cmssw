/** \file
 * Impl of DTWireId
 *
 * \author G. Cerminara - INFN Torino
 * \version $Id: DTWireId.cc,v 1.4 2005/10/24 15:56:19 namapane Exp $
 * \date 02 Aug 2005
*/


#include <iostream>
#include <DataFormats/MuonDetId/interface/DTWireId.h>
#include <FWCore/Utilities/interface/Exception.h>



DTWireId::DTWireId() : DTLayerId() {}



DTWireId::DTWireId(uint32_t id) : DTLayerId(id) {}



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


