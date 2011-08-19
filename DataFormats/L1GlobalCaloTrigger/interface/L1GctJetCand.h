#ifndef L1GCTJETCAND_H
#define L1GCTJETCAND_H

#include <ostream>
#include <string>

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

/// \class L1GctJetCand
/// \brief Level-1 Trigger jet candidate
/// \author Jim Brooke
/// \date June 2006
///


class L1GctJetCand : public L1GctCand {
public:
  /// default constructor (for vector initialisation etc.)
  L1GctJetCand();

  /// construct from raw data - used in GT
  L1GctJetCand(uint16_t rawData, bool isTau, bool isFor);

  /// construct from raw data with source - uesd in GCT unpacker
  L1GctJetCand(uint16_t rawData, bool isTau, bool isFor, uint16_t block, uint16_t index, int16_t bx);

  /// construct from rank, eta, phi, isolation - used in GCT emulator
  /// NB - eta = -6 to -0, +0 to +6. Sign is bit 3, 1 means -ve Z, 0 means +ve Z
  L1GctJetCand(unsigned rank, unsigned phi, unsigned eta, bool isTau, bool isFor);

  /// construct from rank, eta, phi, isolation - will be used in GCT emulator?
  /// NB - eta = -6 to -0, +0 to +6. Sign is bit 3, 1 means -ve Z, 0 means +ve Z
  L1GctJetCand(unsigned rank, unsigned phi, unsigned eta, bool isTau, bool isFor, uint16_t block, uint16_t index, int16_t bx);

  /// destructor (virtual to prevent compiler warnings)
  virtual ~L1GctJetCand();

  /// region associated with the candidate
  L1CaloRegionDetId regionId() const;

  /// name of object
  std::string name() const;

  /// was an object really found?
  bool empty() const;

  /// get the raw data
  uint16_t raw() const { return m_data; }
  
  /// get rank bits
  unsigned rank() const;

  /// get eta index (bit 3 is sign, 1 for -ve Z, 0 for +ve Z)
  unsigned etaIndex() const;

  /// get eta sign bit (1 for -ve Z, 0 for +ve Z)
  unsigned etaSign() const { return (m_data>>9) & 0x1; }
  
  /// get phi index (0-17)
  unsigned phiIndex() const;

  /// check if this is a central jet
  bool isCentral() const { return (!m_isTau) && (!m_isFor); }

  /// check if this is a tau
  bool isTau() const { return m_isTau; }

  /// check if this is a forward jet
  bool isForward() const { return m_isFor; }

  /// which capture block did this come from
  unsigned capBlock() const { return m_captureBlock; }

  /// what index within capture block
  unsigned capIndex() const { return m_captureIndex; }

  /// get bunch-crossing index
  int16_t bx() const { return m_bx; }

  /// equality operator
  int operator==(const L1GctJetCand& c) const { return ((m_data==c.raw() && 
                                                m_isTau==c.isTau() && m_isFor==c.isForward())
                                                 || (this->empty() && c.empty())); }

   /// inequality operator
  int operator!=(const L1GctJetCand& c) const { return !(*this == c); }

private:

  uint16_t m_data;
  bool m_isTau;
  bool m_isFor;
  uint16_t m_captureBlock;
  uint8_t m_captureIndex;
  int16_t m_bx;

 };

std::ostream& operator<<(std::ostream& s, const L1GctJetCand& cand);

#endif
