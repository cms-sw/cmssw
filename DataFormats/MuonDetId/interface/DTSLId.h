#ifndef MuonDetId_DTSLId_H
#define MuonDetId_DTSLId_H

/** \class DTSLId
 *  DetUnit identifier for DT SuperLayers (SL)
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include <DataFormats/MuonDetId/interface/DTChamberId.h>


class DTSLId : public DTChamberId {
public:

  /// Default constructor; fills the common part in the base
  /// and leaves 0 in all other fields
  DTSLId();


  /// Construct from a packed id. It is required that the Detector part of
  /// id is Muon and the SubDet part is DT, otherwise an exception is thrown.
  explicit DTSLId(uint32_t id);


  /// Construct from fully qualified identifier. 
  /// Input values are required to be within legal ranges, otherwise an
  /// exception is thrown.
  DTSLId(int wheel, 
	 int station, 
	 int sector,
	 int superlayer);


  /// superlayer id
  int superlayer() const {
    return ((id_>>slayerStartBit_)&slMask_);
  }


  /// Return the corresponding ChamberId
  DTChamberId chamberId() const {
    return DTChamberId(id_ & chamberIdMask_);
  }

 private:

};

std::ostream& operator<<( std::ostream& os, const DTChamberId& id );

#endif

