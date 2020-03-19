#ifndef L1GlobalTrigger_L1GtPsbWord_h
#define L1GlobalTrigger_L1GtPsbWord_h

/**
 * \class L1GtPsbWord
 * 
 * 
 * Description: PSB block in the L1 GT readout record.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// system include files
#include <iosfwd>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "FWCore/Utilities/interface/typedefs.h"

// forward declarations

// class declaration
class L1GtPsbWord {
public:
  static const int NumberAData = 8;
  static const int NumberBData = 8;

public:
  /// constructors
  L1GtPsbWord();  // empty constructor, all members set to zero;

  /// constructor from unpacked values;
  L1GtPsbWord(cms_uint16_t boardIdValue,
              int bxInEventValue,
              cms_uint16_t bxNrValue,
              cms_uint32_t eventNrValue,
              cms_uint16_t aDataValue[NumberAData],
              cms_uint16_t bDataValue[NumberBData],
              cms_uint16_t localBxNrValue);

  /// destructor
  virtual ~L1GtPsbWord();

  /// equal operator
  bool operator==(const L1GtPsbWord&) const;

  /// unequal operator
  bool operator!=(const L1GtPsbWord&) const;

public:
  /// get/set board ID
  inline const cms_uint16_t boardId() const { return m_boardId; }

  void setBoardId(cms_uint16_t boardIdValue) { m_boardId = boardIdValue; }

  /// set the BoardId value from a 64-bits word, having the index iWord
  /// in the GTFE raw record
  void setBoardId(const cms_uint64_t& word64, int iWord);

  /// set the BoardId value in a 64-bits word, having the index iWord
  /// in the GTFE raw record
  void setBoardIdWord64(cms_uint64_t& word64, int iWord);

  /// get/set bunch cross in the GT event record
  inline const int bxInEvent() const { return m_bxInEvent; }

  void setBxInEvent(int bxInEventValue) { m_bxInEvent = bxInEventValue; }

  /// set the BxInEvent value from a 64-bits word, having the index iWord
  /// in the GTFE raw record
  void setBxInEvent(const cms_uint64_t& word64, int iWord);

  /// set the BxInEvent value in a 64-bits word, having the index iWord
  /// in the GTFE raw record
  void setBxInEventWord64(cms_uint64_t& word64, int iWord);

  /// get/set bunch cross number of the actual bx
  inline const cms_uint16_t bxNr() const { return m_bxNr; }

  void setBxNr(cms_uint16_t bxNrValue) { m_bxNr = bxNrValue; }

  /// set the BxNr value from a 64-bits word, having the index iWord
  /// in the GTFE raw record
  void setBxNr(const cms_uint64_t& word64, int iWord);

  /// set the BxNr value in a 64-bits word, having the index iWord
  /// in the GTFE raw record
  void setBxNrWord64(cms_uint64_t& word64, int iWord);

  /// get/set event number since last L1 reset generated in PSB
  inline const cms_uint32_t eventNr() const { return m_eventNr; }

  void setEventNr(cms_uint32_t eventNrValue) { m_eventNr = eventNrValue; }

  /// set the EventNr value from a 64-bits word, having the index iWord
  /// in the GTFE raw record
  void setEventNr(const cms_uint64_t& word64, int iWord);

  /// set the EventNr value in a 64-bits word, having the index iWord
  /// in the GTFE raw record
  void setEventNrWord64(cms_uint64_t& word64, int iWord);

  /// get/set A_DATA_CH_IA
  const cms_uint16_t aData(int iA) const;
  void setAData(cms_uint16_t aDataVal, int iA);

  /// set the AData value from a 64-bits word, having the index iWord
  /// in the GTFE raw record
  void setAData(const cms_uint64_t& word64, int iWord);

  /// set the AData value in a 64-bits word, having the index iWord
  /// in the GTFE raw record
  void setADataWord64(cms_uint64_t& word64, int iWord);

  /// get/set B_DATA_CH_IB
  const cms_uint16_t bData(int iB) const;
  void setBData(cms_uint16_t bDataVal, int iB);

  /// set the BData value from a 64-bits word, having the index iWord
  /// in the GTFE raw record
  void setBData(const cms_uint64_t& word64, int iWord);

  /// set the BData value in a 64-bits word, having the index iWord
  /// in the GTFE raw record
  void setBDataWord64(cms_uint64_t& word64, int iWord);

  /// get/set local bunch cross number of the actual bx
  inline const cms_uint16_t localBxNr() const { return m_localBxNr; }

  void setLocalBxNr(cms_uint16_t localBxNrValue) { m_localBxNr = localBxNrValue; }

  /// set the local bunch cross number bits from a 64-bits word, having the index iWord
  /// in the GTFE raw record
  void setLocalBxNr(const cms_uint64_t& word64, int iWord);

  /// set the local bunch cross number bits in a 64-bits word, having the index iWord
  /// in the GTFE raw record
  void setLocalBxNrWord64(cms_uint64_t& word64, int iWord);

  /// get the size of the PSB block in GT DAQ record (in multiple of 8 bits)
  inline const unsigned int getSize() const {
    int unitLengthBits = L1GlobalTriggerReadoutSetup::UnitLength;

    return BlockSize * unitLengthBits;
  }

public:
  /// reset the content of a L1GtPsbWord
  void reset();

  /// pretty print
  void print(std::ostream& myCout) const;

private:
  // block description in the raw GT record

  // block size in 64bits words (BlockSize * 64 bits)
  static const int BlockSize = 6;

  // word 0

  // index of the word in the PSB block containig the variable
  static const int BoardIdWord = 0;
  static const int BxInEventWord = 0;
  static const int BxNrWord = 0;
  static const int EventNrWord = 0;

  // mask to get the 64-bit-value from the corresponding word in the PSB block
  static const cms_uint64_t BoardIdMask = 0xFFFF000000000000ULL;
  static const cms_uint64_t BxInEventMask = 0x0000F00000000000ULL;
  static const cms_uint64_t BxNrMask = 0x00000FFF00000000ULL;
  static const cms_uint64_t EventNrMask = 0x0000000000FFFFFFULL;

  // shift to the right to get the value from the "64-bit-value"
  static const int BoardIdShift = 48;
  static const int BxInEventShift = 44;
  static const int BxNrShift = 32;
  static const int EventNrShift = 0;

  // assume uniform size od A_Data and B_Data
  static const int DataCHSize = 16;  // bits
  static const cms_uint64_t DataCHMask = 0x000000000000FFFFULL;

  // word 1
  static const int ADataCH0Word = 1;

  // word 2
  static const int ADataCH4Word = 2;

  // word 3
  static const int BDataCH0Word = 3;

  // word 4
  static const int BDataCH4Word = 4;

  // word 5
  static const int LocalBxNrWord = 5;
  static const cms_uint64_t LocalBxNrMask = 0x0000000000000FFFULL;
  static const int LocalBxNrShift = 0;

private:
  // first number in the comment represents number of bits

  cms_uint16_t m_boardId;  // 16: board identifier
  //
  int m_bxInEvent;  //  4: bunch cross in the GT event record
  //     one keeps 3 bx (F, 0, 1) for standard record,
  //               5 bx (E, F, 0, 1) for debug record
  //
  cms_uint16_t m_bxNr;     // 12: bunch cross number of the actual bx
  cms_uint32_t m_eventNr;  // 24: event number since last L1 reset generated in PSB
  //
  cms_uint16_t m_aData[NumberAData];  // 16: A_Data_ChX
  //
  cms_uint16_t m_bData[NumberBData];  // 16: B_Data_ChX
  //
  cms_uint16_t m_localBxNr;  // 12: local bunch cross number of the actual bx
                             //     bx number at which the data were written into the ringbuffer
};

#endif /*L1GlobalTrigger_L1GtPsbWord_h*/
