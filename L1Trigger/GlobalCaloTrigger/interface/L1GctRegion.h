#ifndef L1GCTREGION_H
#define L1GCTREGION_H

/*!
 * \author Greg Heath
 * \date September 2007
 */

/*! \class L1GctRegion
 * \brief Gct version of a calorimeter region, used within GCT emulation
 * 
 * Only differs from L1CaloRegion by the treatment of overflows
 */

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

class L1GctRegion : public L1CaloRegion
{
 public:

  enum numberOfBits {
    kGctRegionNBits    = 12,
    kGctRegionOFlowBit = 1 << kGctRegionNBits,
    kGctRegionMaxValue = kGctRegionOFlowBit - 1
  };

  L1GctRegion(const unsigned et, const bool overFlow, const bool fineGrain, const unsigned ieta, const unsigned iphi);
  L1GctRegion(const L1CaloRegion&);
  L1GctRegion();

  ~L1GctRegion();

  // Replace et() method to use 12 bits for all eta
  unsigned et() const { return raw()&0x3ff; }

};

#endif
