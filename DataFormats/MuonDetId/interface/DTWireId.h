#ifndef MuonDetId_DTWireId_h
#define MuonDetId_DTWireId_h

/** \class DTWireId
 *  DetUnit identifier for DT wires
 *
 *  $Date: 2005/11/07 16:49:52 $
 *  $Revision: 1.10 $
 *  \author G. Cerminara - INFN Torino
 */

#include <DataFormats/MuonDetId/interface/DTLayerId.h>


class DTWireId :public DTLayerId {
 public:
      
  /// Default constructor; fills the common part in the base
  /// and leaves 0 in all other fields
  DTWireId();


  /// Construct from a packed id. It is required that the Detector part of
  /// id is Muon and the SubDet part is DT, otherwise an exception is thrown.
  explicit DTWireId(uint32_t id);


  /// Construct from fully qualified identifier. 
  /// Input values are required to be within legal ranges, otherwise an
  /// exception is thrown.
  DTWireId(int wheel, 
	   int station, 
	   int sector,
	   int superlayer,
	   int layer,
	   int wire);
  
  /// wire id
  int wire() const {
    return ((id_>>wireStartBit_)&wireMask_);
  }

  /// Return the corresponding LayerId
  DTLayerId layerId() const {
    return DTLayerId(id_ & layerIdMask_);
  }
  

 private:
 

};

std::ostream& operator<<( std::ostream& os, const DTWireId& id );

#endif
