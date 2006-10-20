#ifndef L1GCTCAND_H
#define L1GCTCAND_H

#include "DatFormats/L1CaloTrigger/interface/LCaloRegionDetId.h"

/// \class L1GctCand
/// \brief ABC for GCT EM and jet candidates
/// \author Jim Brooke
/// \date July 2006
///

class L1GctCand {
public:

  /// access origin of candidate
  virtual L1CaloRegionDetId regionId() const = 0;

  /// empty candidate  - true if object not initialized
  virtual bool empty() const = 0;

  /// \brief get the rank code (6 bits)
  ///
  /// Note that the precise meaning of the bits returned by this method
  /// (ie. the Et scale) may differ depending on the concrete type
  virtual unsigned rank() const = 0; 

  /// get eta index (bit 3 is sign, 1 for -ve Z, 0 for +ve Z)
  virtual unsigned etaIndex() const = 0;

  /// get eta sign bit (1 for -ve Z, 0 for +ve Z)
  virtual unsigned etaSign() const = 0;
  
  /// get phi index (0-17)
  virtual unsigned phiIndex() const = 0;




};


#endif
