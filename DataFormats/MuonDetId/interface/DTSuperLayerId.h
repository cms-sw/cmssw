#ifndef MuonDetId_DTSuperLayerId_H
#define MuonDetId_DTSuperLayerId_H

/** \class DTSuperLayerId
 *  DetUnit identifier for DT SuperLayers (SL)
 *
 *  $Date: 2006/01/19 15:41:32 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include <DataFormats/MuonDetId/interface/DTChamberId.h>


class DTSuperLayerId : public DTChamberId {
public:

  /// Default constructor; fills the common part in the base
  /// and leaves 0 in all other fields
  DTSuperLayerId();


  /// Construct from a packed id. It is required that the Detector part of
  /// id is Muon and the SubDet part is DT, otherwise an exception is thrown.
  explicit DTSuperLayerId(uint32_t id);


  /// Construct from fully qualified identifier. 
  /// Input values are required to be within legal ranges, otherwise an
  /// exception is thrown.
  DTSuperLayerId(int wheel, 
		 int station, 
		 int sector,
		 int superlayer);


  /// Copy Constructor.
  /// It takes care of masking fields which are not meaningful for a DTSuperLayerId
  DTSuperLayerId(const DTSuperLayerId& slId);


  /// Constructor from a camberId and SL number
  DTSuperLayerId(const DTChamberId& chId, int superlayer);


  /// Return the superlayer number
  int superlayer() const {
    return ((id_>>slayerStartBit_)&slMask_);
  }


  /// Return the corresponding ChamberId
  DTChamberId chamberId() const {
    return DTChamberId(id_ & chamberIdMask_);
  }

 private:

};

std::ostream& operator<<( std::ostream& os, const DTSuperLayerId& id );

#endif

