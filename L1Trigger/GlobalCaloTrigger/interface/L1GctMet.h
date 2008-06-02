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

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

class L1GctMet
{
 public:

  enum metAlgoType { cordicTranslate, oldGct, floatingPoint };

  typedef L1GctUnsignedInt<  L1GctEtMiss::kEtMissNBits    > etMissType;
  typedef L1GctUnsignedInt<  L1GctEtMiss::kEtMissPhiNBits > etMissPhiType;
  typedef L1GctJetLeafCard::etComponentType etComponentType;

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

 private:

  struct etmiss_internal {
    unsigned mag;
    unsigned phi;
  };

  etComponentType m_exComponent;
  etComponentType m_eyComponent;
  metAlgoType m_algoType;

  etmiss_internal cordicTranslateAlgo (const int ex, const int ey) const;
  etmiss_internal oldGctAlgo          (const int ex, const int ey) const;
  etmiss_internal floatingPointAlgo   (const int ex, const int ey) const;

  int cordicShiftAndRoundBits (const int e, const unsigned nBits) const;
};

#endif
