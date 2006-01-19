#ifndef MuonDetId_DTLayerId_h
#define MuonDetId_DTLayerId_h

/** \class DTLayerId
 *  DetUnit identifier for DT layers
 *
 *  $Date: 2005/12/19 16:15:11 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include <DataFormats/MuonDetId/interface/DTSuperLayerId.h>

class DTLayerId : public DTSuperLayerId {
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


  /// Copy Constructor.
  /// It takes care of masking fields which are not meaningful for a DTLayerId
  DTLayerId(const DTLayerId& layerId);


  /// Constructor from a camberId and SL and layer numbers
  DTLayerId(const DTChamberId& chId, int superlayer, int layer);


  /// Constructor from a SuperLayerId and layer number
  DTLayerId(const DTSuperLayerId& slId, int layer);


  /// Return the layer number
  int layer() const {
    return ((id_>>layerStartBit_)&lMask_);
  }


  /// Return the corresponding SuperLayerId
  DTSuperLayerId superlayerId() const {
    return DTSuperLayerId(id_ & slIdMask_);
  }
  


};


std::ostream& operator<<( std::ostream& os, const DTLayerId& id );

#endif
