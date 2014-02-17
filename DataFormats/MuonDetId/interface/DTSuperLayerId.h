#ifndef MuonDetId_DTSuperLayerId_H
#define MuonDetId_DTSuperLayerId_H

/** \class DTSuperLayerId
 *  DetUnit identifier for DT SuperLayers (SL)
 *
 *  $Date: 2006/04/12 17:52:39 $
 *  $Revision: 1.4 $
 *  \author G. Cerminara - INFN Torino
 */

#include <DataFormats/MuonDetId/interface/DTChamberId.h>


class DTSuperLayerId : public DTChamberId {
public:

  /// Default constructor. It fills the common part in the base
  /// and leaves 0 in all other fields
  DTSuperLayerId();


  /// Construct from a packed id.
  /// It is required that the packed id represents a valid DT DetId
  /// (proper Detector and  SubDet fields), otherwise an exception is thrown.
  /// Any bits outside the DTSuperLayerId fields are zeroed; apart for
  /// this, no check is done on the vaildity of the values.
  explicit DTSuperLayerId(uint32_t id);


  /// Construct from indexes.
  /// Input values are required to be within legal ranges, otherwise an
  /// exception is thrown.
  DTSuperLayerId(int wheel, 
		 int station, 
		 int sector,
		 int superlayer);


  /// Copy Constructor.
  /// Any bits outside the DTChamberId fields are zeroed; apart for
  /// this, no check is done on the vaildity of the values.
  DTSuperLayerId(const DTSuperLayerId& slId);


  /// Constructor from a DTChamberId and SL number.
  DTSuperLayerId(const DTChamberId& chId, int superlayer);


  /// Return the superlayer number
  int superLayer() const {
    return ((id_>>slayerStartBit_)&slMask_);
  }


  /// Return the superlayer number (deprecated method name)
  int superlayer() const {
    return superLayer();
  }


  /// Return the corresponding ChamberId
  DTChamberId chamberId() const {
    return DTChamberId(id_ & chamberIdMask_);
  }

 private:

};

std::ostream& operator<<( std::ostream& os, const DTSuperLayerId& id );

#endif

