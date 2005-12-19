#ifndef MuonDetId_DTLayerId_h
#define MuonDetId_DTLayerId_h

/** \class DTLayerId
 *  DetUnit identifier for DT layers
 *
 *  $Date: 2005/11/07 16:49:52 $
 *  $Revision: 1.10 $
 *  \author G. Cerminara - INFN Torino
 */

#include <DataFormats/MuonDetId/interface/DTSLId.h>

class DTLayerId : public DTSLId {
 public:
      
  /// Default constructor; fills the common part in the base
  /// and leaves 0 in all other fields
  DTLayerId();


  /// Construct from a packed id. It is required that the Detector part of
  /// id is Muon and the SubDet part is DT, otherwise an exception is thrown.
  explicit DTLayerId(uint32_t id);


  /// Construct from fully qualified identifier.
  /// Input values are required to be within legal ranges, otherwise an
  /// exception is thrown.
  DTLayerId(int wheel, 
	    int station, 
	    int sector,
	    int superlayer,
	    int layer);



  /// layer id
  int layer() const {
    return ((id_>>layerStartBit_)&lMask_);
  }


  /// Return the corresponding SuperLayerId
  DTSLId superlayerId() const {
    return DTSLId(id_ & slIdMask_);
  }
  


};


std::ostream& operator<<( std::ostream& os, const DTLayerId& id );

#endif
