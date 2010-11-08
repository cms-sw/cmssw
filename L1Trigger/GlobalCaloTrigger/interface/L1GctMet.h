#ifndef L1GCTMET_H
#define L1GCTMET_H

/*!
 * \author Greg Heath
 * \date April 2008
 */

/*! \class L1GctMet
 * \brief Stores Level-1 missing Et in (Ex, Ey) form, allowing it to be retrieved as (magnitude, angle)
 * 
 * Allows the implementation of alternative algorithms
 */

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelEnergyFpga.h"

class L1CaloEtScale;
class L1GctHtMissLut;

class L1GctMet
{
 public:

  enum metAlgoType { cordicTranslate, useHtMissLut, oldGct, floatingPoint };

  typedef L1GctUnsignedInt<  L1GctEtMiss::kEtMissNBits    > etMissType;
  typedef L1GctUnsignedInt<  L1GctEtMiss::kEtMissPhiNBits > etMissPhiType;
  typedef L1GctWheelEnergyFpga::etComponentType etComponentType;

  struct etmiss_vec {
    etMissType    mag;
    etMissPhiType phi;
  };

  L1GctMet(const unsigned ex=0, const unsigned ey=0, const metAlgoType algo=cordicTranslate);
  L1GctMet(const etComponentType& ex, const etComponentType& ey, const metAlgoType algo=cordicTranslate);
  ~L1GctMet();

  // return the missing Et as (magnitude, angle)
  etmiss_vec metVector() const ;

  // set and get the components
  void setComponents(const unsigned ex, const unsigned ey) { setExComponent(ex); setEyComponent(ey); }
  void setComponents(const etComponentType& ex, const etComponentType& ey) { setExComponent(ex); setEyComponent(ey); }
  void setExComponent(const unsigned ex);
  void setEyComponent(const unsigned ey);
  void setExComponent(const etComponentType& ex) { m_exComponent = ex; }
  void setEyComponent(const etComponentType& ey) { m_eyComponent = ey; }
  etComponentType getExComponent() const { return m_exComponent; }
  etComponentType getEyComponent() const { return m_eyComponent; }

  // set and get the algorithm type
  void setAlgoType(const metAlgoType algo) { m_algoType = algo; }
  metAlgoType getAlgoType() const { return m_algoType; }

  // set and get the bit shift
  // This parameter can be used to scale the output relative to the input
  void setBitShift(const unsigned nbits) { m_bitShift = nbits; }
  unsigned getBitShift() const { return m_bitShift; }

  // get the LUT (used by L1GctPrintLuts)
  const L1GctHtMissLut* getHtMissLut() const { return m_htMissLut; }

  // set and get the LUT parameters
  void setEtScale(const L1CaloEtScale* const fn);
  void setEtComponentLsb(const double lsb);

  const L1CaloEtScale* etScale() const;
  const double componentLsb() const;
 private:

  enum etComponentShift { kExOrEyMissComponentShift=4 };

  struct etmiss_internal {
    unsigned mag;
    unsigned phi;
  };


  /// Private method to check for an overflow condition on the input components
  /// Allows the check to depend on the algorithm type
  const bool inputOverFlow() const;

  etComponentType m_exComponent;
  etComponentType m_eyComponent;
  metAlgoType     m_algoType;
  unsigned short  m_bitShift;

  L1GctHtMissLut* m_htMissLut;

  etmiss_internal cordicTranslateAlgo (const int ex, const int ey) const;
  etmiss_internal useHtMissLutAlgo    (const int ex, const int ey) const;
  etmiss_internal oldGctAlgo          (const int ex, const int ey) const;
  etmiss_internal floatingPointAlgo   (const int ex, const int ey) const;

  int cordicShiftAndRoundBits (const int e, const unsigned nBits) const;
};

#endif
