#ifndef L1GCTEMCAND_H
#define L1GCTEMCAND_H

#include <ostream>
#include <string>

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"

/*! \class L1GctEmCand
 * \brief Level-1 Trigger EM candidate at output of GCT
 *
 */

/*! \author Jim Brooke
 *  \date June 2006
 */


class L1GctEmCand : public L1GctCand {
public:

  /// default constructor (for vector initialisation etc.)
  L1GctEmCand();

  /// construct from raw data, no source - used in GT
  L1GctEmCand(uint16_t rawData, bool iso);

  /// construct from raw data with source - used in GCT unpacker
  L1GctEmCand(uint16_t rawData, bool iso, uint16_t block, uint16_t index, int16_t bx);

  /// construct from rank, eta, phi, isolation - used in GCT emulator
  /// eta = -6 to -0, +0 to +6. Sign is bit 3, 1 means -ve Z, 0 means +ve Z
  L1GctEmCand(unsigned rank, unsigned phi, unsigned eta, bool iso);

  /// construct from rank, eta, phi, isolation - could be used in GCT emulator?
  /// eta = -6 to -0, +0 to +6. Sign is bit 3, 1 means -ve Z, 0 means +ve Z
  L1GctEmCand(unsigned rank, unsigned phi, unsigned eta, bool iso, uint16_t block, uint16_t index, int16_t bx);

  /// construct from RCT output candidate
  L1GctEmCand(L1CaloEmCand& c);

  /// destructor (virtual to prevent compiler warnings)
  ~L1GctEmCand() override;
  
  /// region associated with the candidate
  L1CaloRegionDetId regionId() const override;

  /// name of object
  std::string name() const;

  /// was an object really found?
  bool empty() const override  {  return (rank() == 0); }
 
  /// get the raw data
  uint16_t raw() const { return m_data; }
  
  /// get rank bits
  unsigned rank() const override  { return m_data & 0x3f; }

  /// get eta index -6 to -0, +0 to +6 (bit 3 is sign, 1 for -ve Z, 0 for +ve Z)
  unsigned etaIndex() const override  { return (m_data>>6) & 0xf; } 

  /// get eta sign (1 for -ve Z, 0 for +ve Z) 
  unsigned etaSign() const override { return (m_data>>9) & 0x1; } 

  /// get phi index (0-17)
  unsigned phiIndex() const override  { return (m_data>>10) & 0x1f; } 

  /// which stream did this come from
  bool isolated() const { return m_iso; }

  /// which capture block did this come from
  unsigned capBlock() const { return m_captureBlock; }

  /// what index within capture block
  unsigned capIndex() const { return m_captureIndex; }

  /// get bunch-crossing index
  int16_t bx() const { return m_bx; }

  /// equality operator
  int operator==(const L1GctEmCand& c) const { return ((m_data==c.raw() && m_iso==c.isolated())
                                                      || (this->empty() && c.empty())); }

  /// inequality operator
  int operator!=(const L1GctEmCand& c) const { return !(*this == c); }

 private:

  // set internal data from rank and region ieta, iphi
  void construct(unsigned rank, unsigned eta, unsigned phi);

 private:

  uint16_t m_data;
  bool m_iso;
  uint16_t m_captureBlock;
  uint8_t m_captureIndex;
  int16_t m_bx;

 };


std::ostream& operator<<(std::ostream& s, const L1GctEmCand& cand);



#endif 
