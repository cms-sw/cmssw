#ifndef CondFormats_L1TObjects_L1GtBoard_h
#define CondFormats_L1TObjects_L1GtBoard_h

/**
 * \class L1GtBoard
 *
 *
 * Description: class for L1 GT board.
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

#include <string>
#include <vector>
#include <map>

// user include files
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include <cstdint>

// forward declarations

// class declaration
class L1GtBoard {
public:
  /// constructors
  L1GtBoard();

  L1GtBoard(const L1GtBoardType&);

  L1GtBoard(const L1GtBoardType&, const int&);

  /// destructor
  virtual ~L1GtBoard();

  /// copy constructor
  L1GtBoard(const L1GtBoard&);

  /// assignment operator
  L1GtBoard& operator=(const L1GtBoard&);

  /// equal operator
  bool operator==(const L1GtBoard&) const;

  /// unequal operator
  bool operator!=(const L1GtBoard&) const;

  /// less than operator
  bool operator<(const L1GtBoard&) const;

public:
  /// number of InfiniCables per board
  static const int NumberCablesBoard;

public:
  /// get / set board type
  inline const L1GtBoardType gtBoardType() const { return m_gtBoardType; }

  void setGtBoardType(const L1GtBoardType&);

  /// get / set board index
  inline const int gtBoardIndex() const { return m_gtBoardIndex; }

  void setGtBoardIndex(const int&);

  /// get / set the position of board data block
  /// in the GT DAQ readout record
  inline const int gtPositionDaqRecord() const { return m_gtPositionDaqRecord; }

  void setGtPositionDaqRecord(const int&);

  /// get / set the position of board data block
  /// in the GT EVM readout record
  inline const int gtPositionEvmRecord() const { return m_gtPositionEvmRecord; }

  void setGtPositionEvmRecord(const int&);

  /// get / set the bit of board in the GTFE ACTIVE_BOARDS
  /// for the GT DAQ readout record
  inline const int gtBitDaqActiveBoards() const { return m_gtBitDaqActiveBoards; }

  void setGtBitDaqActiveBoards(const int&);

  /// get / set the bit of board in the GTFE ACTIVE_BOARDS
  /// for the GT EVM readout record
  inline const int gtBitEvmActiveBoards() const { return m_gtBitEvmActiveBoards; }

  void setGtBitEvmActiveBoards(const int&);

  /// get / set board slot
  inline const int gtBoardSlot() const { return m_gtBoardSlot; }

  void setGtBoardSlot(const int&);

  /// get / set board hex fragment name in hw record
  inline const int gtBoardHexName() const { return m_gtBoardHexName; }

  void setGtBoardHexName(const int&);

  /// get / set L1 quadruplet (4x16 bits)(cable) in the PSB input
  /// valid for PSB only
  const std::vector<L1GtPsbQuad>& gtQuadInPsb() const { return m_gtQuadInPsb; }

  void setGtQuadInPsb(const std::vector<L1GtPsbQuad>&);

  /// get / set detailed input configuration for PSB (objects pro channel)
  /// int: channel number
  const std::map<int, std::vector<L1GtObject> >& gtInputPsbChannels() const { return m_gtInputPsbChannels; }

  void setGtInputPsbChannels(const std::map<int, std::vector<L1GtObject> >&);

  /// get the board ID
  const uint16_t gtBoardId() const;

  /// return board name - it depends on L1GtBoardType enum!!!
  std::string gtBoardName() const;

  /// print board
  void print(std::ostream& myCout) const;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const L1GtBoard&);

private:
  /// board type (from L1GtBoardType enum)
  L1GtBoardType m_gtBoardType;

  /// board index
  int m_gtBoardIndex;

  /// the position of board data block in the GT DAQ readout record
  int m_gtPositionDaqRecord;

  /// the position of board data block in the GT EVM readout record
  int m_gtPositionEvmRecord;

  /// the bit of board in the GTFE ACTIVE_BOARDS
  /// for the GT DAQ readout record
  int m_gtBitDaqActiveBoards;

  /// the bit of board in the GTFE ACTIVE_BOARDS
  /// for the GT EVM readout record
  int m_gtBitEvmActiveBoards;

  /// the slot of board (part of Board_Id)
  int m_gtBoardSlot;

  /// board hex fragment name in hw record
  /// gives the bits written for each GT board in the Board_Id
  int m_gtBoardHexName;

  /// valid for PSB only
  /// L1 quadruplet (4x16 bits)(cable) in the PSB input
  std::vector<L1GtPsbQuad> m_gtQuadInPsb;

  /// valid for PSB only
  /// detailed input configuration for PSB (objects pro channel)
  /// int: channel number
  std::map<int, std::vector<L1GtObject> > m_gtInputPsbChannels;

  COND_SERIALIZABLE;
};

#endif /*CondFormats_L1TObjects_L1GtBoard_h*/
