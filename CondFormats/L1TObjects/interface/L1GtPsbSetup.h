#ifndef CondFormats_L1TObjects_L1GtPsbSetup_h
#define CondFormats_L1TObjects_L1GtPsbSetup_h

/**
 * \class L1GtPsbSetup
 *
 *
 * Description: setup for L1 GT PSB boards.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// system include files
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <iosfwd>

// user include files
#include "CondFormats/L1TObjects/interface/L1GtPsbConfig.h"

// forward declarations

// class declaration
class L1GtPsbSetup {
public:
  // constructor
  L1GtPsbSetup();

  // destructor
  virtual ~L1GtPsbSetup();

public:
  /// get / set / print the setup for L1 GT PSB boards
  const std::vector<L1GtPsbConfig>& gtPsbSetup() const { return m_gtPsbSetup; }

  void setGtPsbSetup(const std::vector<L1GtPsbConfig>&);

  void print(std::ostream&) const;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const L1GtPsbSetup&);

private:
  /// L1 GT PSB boards and their setup
  std::vector<L1GtPsbConfig> m_gtPsbSetup;

  COND_SERIALIZABLE;
};

#endif /*CondFormats_L1TObjects_L1GtPsbSetup_h*/
