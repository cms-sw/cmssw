#ifndef L1GCTEMCAND_H
#define L1GCTEMCAND_H

#include <boost/cstdint.hpp>
#include <ostream>
#include <string>

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"

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

  /// construct from raw data
  L1GctEmCand(uint16_t data, bool iso);

  /// construct from rank, eta, phi, isolation
  /// eta = -6 to -0, +0 to +6. Sign is bit 3, 1 means -ve Z, 0 means +ve Z
  L1GctEmCand(unsigned rank, unsigned phi, unsigned eta, bool iso);

   /// destructor (virtual to prevent compiler warnings)
  virtual ~L1GctEmCand();
  
  /// region associated with the candidate
  L1CaloRegionDetId regionId() const { return id; }

  /// name of object
  std::string name() const;

  /// was an object really found?
  bool empty() const;
  
  /// get the raw data
  uint16_t raw() const { return m_data; }
  
  /// get rank bits
  unsigned rank() const { return m_data & 0x3f; }

  /// get eta index -6 to -0, +0 to +6 (bit 3 is sign, 1 for -ve Z, 0 for +ve Z)
  unsigned etaIndex() const { return (m_data>>6) & 0xf; }

  /// get eta sign (1 for -ve Z, 0 for +ve Z)
  unsigned etaSign() const { return (m_data>>9) & 0x1; }

  /// get phi index (0-17)
  unsigned phiIndex() const { return (m_data>>10) & 0x1f; }

  /// which stream did this come from
  bool isolated() const { return m_iso; }

 private:

  L1CaloRegionDetId m_id;

  uint16_t m_data;
  bool m_iso;

 };


std::ostream& operator<<(std::ostream& s, const L1GctEmCand& cand);



#endif 
