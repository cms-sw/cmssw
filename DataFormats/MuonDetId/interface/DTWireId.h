#ifndef MuonDetId_DTWireId_h
#define MuonDetId_DTWireId_h

/** \class DTWireId
 *  DetUnit identifier for DT wires
 *
 *  $Date: 2006/04/12 17:52:39 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - INFN Torino
 */

#include <DataFormats/MuonDetId/interface/DTLayerId.h>


class DTWireId :public DTLayerId {
 public:
      
  /// Default constructor.
  /// Fills the common part in the base and leaves 0 in all other fields
  DTWireId();


  /// Construct from a packed id. 
  /// It is required that the packed id represents a valid DT DetId
  /// (proper Detector and  SubDet fields), otherwise an exception is thrown.
  /// No check is done on the vaildity of the values.
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
  

  /// Copy Constructor.
  DTWireId(const DTWireId& wireId);


  /// Constructor from a CamberId and SL, layer and wire numbers
  DTWireId(const DTChamberId& chId, int superlayer, int layer, int wire);


  /// Constructor from a SuperLayerId and layer and wire numbers
  DTWireId(const DTSuperLayerId& slId, int layer, int wire);


  /// Constructor from a layerId and a wire number
  DTWireId(const DTLayerId& layerId, int wire);


  /// Return the wire number
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
