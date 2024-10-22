#ifndef CondFormats_L1TObjects_L1GtParameters_h
#define CondFormats_L1TObjects_L1GtParameters_h

/**
 * \class L1GtParameters
 *
 *
 * Description: L1 GT parameters.
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

#include <ostream>
#include <vector>
#include <cstdint>

// user include files
//   base class

// forward declarations

// class declaration
class L1GtParameters {
public:
  // constructor
  L1GtParameters();

  // destructor
  virtual ~L1GtParameters();

public:
  /// get / set the total Bx's in the event
  inline const int gtTotalBxInEvent() const { return m_totalBxInEvent; }

  void setGtTotalBxInEvent(const int&);

  /// get / set the active boards for L1 GT DAQ record
  inline const uint16_t gtDaqActiveBoards() const { return m_daqActiveBoards; }

  void setGtDaqActiveBoards(const uint16_t&);

  /// get / set the active boards for L1 GT EVM record
  inline const uint16_t gtEvmActiveBoards() const { return m_evmActiveBoards; }

  void setGtEvmActiveBoards(const uint16_t&);

  /// get / set the number of Bx per board for L1 GT DAQ record
  inline const std::vector<int>& gtDaqNrBxBoard() const { return m_daqNrBxBoard; }

  void setGtDaqNrBxBoard(const std::vector<int>&);

  /// get / set the number of Bx per board for L1 GT EVM record
  inline const std::vector<int>& gtEvmNrBxBoard() const { return m_evmNrBxBoard; }

  void setGtEvmNrBxBoard(const std::vector<int>&);

  /// get / set length of BST message (in bytes) for L1 GT EVM record
  inline const unsigned int gtBstLengthBytes() const { return m_bstLengthBytes; }

  void setGtBstLengthBytes(const unsigned int&);

  /// print all the L1 GT parameters
  void print(std::ostream&) const;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const L1GtParameters&);

private:
  /// total Bx's in the event
  int m_totalBxInEvent;

  /// active boards in the L1 DAQ record
  uint16_t m_daqActiveBoards;

  /// active boards in the L1 EVM record
  uint16_t m_evmActiveBoards;

  /// number of Bx per board in the DAQ record
  std::vector<int> m_daqNrBxBoard;

  /// number of Bx per board in the EVM record
  std::vector<int> m_evmNrBxBoard;

  /// length of BST message (in bytes)
  unsigned int m_bstLengthBytes;

  COND_SERIALIZABLE;
};

#endif /*CondFormats_L1TObjects_L1GtParameters_h*/
