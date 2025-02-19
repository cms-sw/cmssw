#ifndef MuonDetId_DTLayerId_h
#define MuonDetId_DTLayerId_h

/** \class DTLayerId
 *  DetUnit identifier for DT layers
 *
 *  $Date: 2006/04/12 17:52:39 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - INFN Torino
 */

#include <DataFormats/MuonDetId/interface/DTSuperLayerId.h>

class DTLayerId : public DTSuperLayerId {
 public:
      
  /// Default constructor. 
  /// Fills the common part in the base and leaves 0 in all other fields
  DTLayerId();


  /// Construct from a packed id.   
  /// It is required that the packed id represents a valid DT DetId
  /// (proper Detector and  SubDet fields), otherwise an exception is thrown.
  /// Any bits outside the DTLayerId fields are zeroed; apart for
  /// this, no check is done on the vaildity of the values.
  explicit DTLayerId(uint32_t id);


  /// Construct from indexes.
  /// Input values are required to be within legal ranges, otherwise an
  /// exception is thrown.
  DTLayerId(int wheel, 
	    int station, 
	    int sector,
	    int superlayer,
	    int layer);


  /// Copy Constructor.
  /// Any bits outside the DTLayerId fields are zeroed; apart for
  /// this, no check is done on the vaildity of the values.
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
