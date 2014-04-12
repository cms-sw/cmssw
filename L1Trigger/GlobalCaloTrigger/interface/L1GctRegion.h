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
    kGctRegionNBits    = 10,
    kGctRegionOFlowBit = 1 << kGctRegionNBits,
    kGctRegionMaxValue = kGctRegionOFlowBit - 1
  };

  // Constructors and destructor
  L1GctRegion();

  ~L1GctRegion();

  // Named constructors
  static L1GctRegion makeJfInputRegion(const L1CaloRegion&);
  static L1GctRegion makeProtoJetRegion(const unsigned et,   const bool overFlow, const bool fineGrain, const bool tauIsolationVeto,
                                        const unsigned ieta, const unsigned iphi, const int16_t bx);
  static L1GctRegion makeFinalJetRegion(const unsigned et,   const bool overFlow, const bool fineGrain,
                                        const unsigned ieta, const unsigned iphi, const int16_t bx);

  // Replace et() method to use 10 bits for all eta
  unsigned et() const { return overFlow() ? kGctRegionMaxValue : raw()&kGctRegionMaxValue; }

  // Replace local eta with a non-physical value
  unsigned rctEta() const { return ( empty() ? 12 : id().rctEta() ); }

  // Access to additional bit fields
  bool featureBit0() { return ((raw() >> 14) & 0x1) != 0; } 
  bool featureBit1() { return ((raw() >> 15) & 0x1) != 0; }

  void setFeatureBit0() { setBit(14, true ); }
  void clrFeatureBit0() { setBit(14, false); }
  void setFeatureBit1() { setBit(15, true ); }
  void clrFeatureBit1() { setBit(15, false); }

 private:

  // constructor for internal use
  L1GctRegion(const unsigned et,
	      const bool overFlow, 
	      const bool fineGrain,
	      const unsigned ieta, 
	      const unsigned iphi,
	      const int16_t bx);


  void setBit(const unsigned bitNum, const bool onOff);

};

#endif
