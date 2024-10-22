/**
 * \class L1GtfeWord
 *
 *
 * Description: L1 Global Trigger - GTFE words in the readout record.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"

// system include files
#include <iomanip>

// user include files

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors

// empty constructor, all members set to zero;
L1GtfeWord::L1GtfeWord()
    : m_boardId(0),
      m_recordLength1(0),
      m_recordLength(0),
      m_bxNr(0),
      m_setupVersion(0),
      m_activeBoards(0),
      m_altNrBxBoard(0),
      m_totalTriggerNr(0) {
  // empty
}

// constructor from unpacked values;
L1GtfeWord::L1GtfeWord(cms_uint16_t boardIdValue,
                       cms_uint16_t recordLength1Value,
                       cms_uint16_t recordLengthValue,
                       cms_uint16_t bxNrValue,
                       cms_uint32_t setupVersionValue,
                       cms_uint16_t activeBoardsValue,
                       cms_uint16_t altNrBxBoardValue,
                       cms_uint32_t totalTriggerNrValue)
    : m_boardId(boardIdValue),
      m_recordLength1(recordLength1Value),
      m_recordLength(recordLengthValue),
      m_bxNr(bxNrValue),
      m_setupVersion(setupVersionValue),
      m_activeBoards(activeBoardsValue),
      m_altNrBxBoard(altNrBxBoardValue),
      m_totalTriggerNr(totalTriggerNrValue)

{
  // empty
}

// destructor
L1GtfeWord::~L1GtfeWord() {
  // empty now
}

// equal operator
bool L1GtfeWord::operator==(const L1GtfeWord& result) const {
  if (m_boardId != result.m_boardId) {
    return false;
  }

  if (m_recordLength1 != result.m_recordLength1) {
    return false;
  }

  if (m_recordLength != result.m_recordLength) {
    return false;
  }

  if (m_bxNr != result.m_bxNr) {
    return false;
  }

  if (m_setupVersion != result.m_setupVersion) {
    return false;
  }

  if (m_activeBoards != result.m_activeBoards) {
    return false;
  }

  if (m_altNrBxBoard != result.m_altNrBxBoard) {
    return false;
  }

  if (m_totalTriggerNr != result.m_totalTriggerNr) {
    return false;
  }

  // all members identical
  return true;
}

// unequal operator
bool L1GtfeWord::operator!=(const L1GtfeWord& result) const { return !(result == *this); }

// methods

// set the BoardId value from a 64-bits word
void L1GtfeWord::setBoardId(const cms_uint64_t& word64) { m_boardId = (word64 & BoardIdMask) >> BoardIdShift; }

// set the BoardId value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setBoardIdWord64(cms_uint64_t& word64, int iWord) {
  if (iWord == BoardIdWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_boardId) << BoardIdShift);
  }
}

// set the RecordLength1 value from a 64-bits word
void L1GtfeWord::setRecordLength1(const cms_uint64_t& word64) {
  m_recordLength1 = (word64 & RecordLength1Mask) >> RecordLength1Shift;
}

// set the RecordLength1 value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setRecordLength1Word64(cms_uint64_t& word64, int iWord) {
  if (iWord == RecordLength1Word) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_recordLength1) << RecordLength1Shift);
  }
}

// set the RecordLength value from a 64-bits word
void L1GtfeWord::setRecordLength(const cms_uint64_t& word64) {
  m_recordLength = (word64 & RecordLengthMask) >> RecordLengthShift;
}

// set the RecordLength value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setRecordLengthWord64(cms_uint64_t& word64, int iWord) {
  if (iWord == RecordLengthWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_recordLength) << RecordLengthShift);
  }
}

// set the BxNr value from a 64-bits word
void L1GtfeWord::setBxNr(const cms_uint64_t& word64) { m_bxNr = (word64 & BxNrMask) >> BxNrShift; }

// set the BxNr value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setBxNrWord64(cms_uint64_t& word64, int iWord) {
  if (iWord == BxNrWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_bxNr) << BxNrShift);
  }
}

// set the SetupVersion value from a 64-bits word
void L1GtfeWord::setSetupVersion(const cms_uint64_t& word64) {
  m_setupVersion = (word64 & SetupVersionMask) >> SetupVersionShift;
}

// set the SetupVersion value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setSetupVersionWord64(cms_uint64_t& word64, int iWord) {
  if (iWord == SetupVersionWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_setupVersion) << SetupVersionShift);
  }
}

// get / set BST flag: 0 or 1 - via setup version (no private member)
const int L1GtfeWord::bstFlag() const {
  int bstFlagValue = 0;
  bstFlagValue = static_cast<int>(m_setupVersion & BstFlagMask);

  return bstFlagValue;
}

void L1GtfeWord::setBstFlag(const int bstFlagValue) {
  m_setupVersion = m_setupVersion | (static_cast<cms_uint32_t>(bstFlagValue) & BstFlagMask);
}

// set the ActiveBoards value from a 64-bits word
void L1GtfeWord::setActiveBoards(const cms_uint64_t& word64) {
  m_activeBoards = (word64 & ActiveBoardsMask) >> ActiveBoardsShift;
}

// set the ActiveBoards value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setActiveBoardsWord64(cms_uint64_t& word64, int iWord) {
  if (iWord == ActiveBoardsWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_activeBoards) << ActiveBoardsShift);
  }
}

// set the ActiveBoards value in a 64-bits word, having the index iWord
// in the GTFE raw record from the value activeBoardsValue
void L1GtfeWord::setActiveBoardsWord64(cms_uint64_t& word64, int iWord, cms_int16_t activeBoardsValue) {
  if (iWord == ActiveBoardsWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(activeBoardsValue) << ActiveBoardsShift);
  }
}

// set the AltNrBxBoard value from a 64-bits word
void L1GtfeWord::setAltNrBxBoard(const cms_uint64_t& word64) {
  m_altNrBxBoard = (word64 & AltNrBxBoardMask) >> AltNrBxBoardShift;
}

// set the AltNrBxBoard value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setAltNrBxBoardWord64(cms_uint64_t& word64, int iWord) {
  if (iWord == AltNrBxBoardWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_altNrBxBoard) << AltNrBxBoardShift);
  }
}

// set the AltNrBxBoard value in a 64-bits word, having the index iWord
// in the GTFE raw record from the value altNrBxBoardValue
void L1GtfeWord::setAltNrBxBoardWord64(cms_uint64_t& word64, int iWord, cms_int16_t altNrBxBoardValue) {
  if (iWord == AltNrBxBoardWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(altNrBxBoardValue) << AltNrBxBoardShift);
  }
}

// set the TotalTriggerNr value from a 64-bits word
void L1GtfeWord::setTotalTriggerNr(const cms_uint64_t& word64) {
  m_totalTriggerNr = (word64 & TotalTriggerNrMask) >> TotalTriggerNrShift;
}

// set the TotalTriggerNr value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setTotalTriggerNrWord64(cms_uint64_t& word64, int iWord) {
  if (iWord == TotalTriggerNrWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_totalTriggerNr) << TotalTriggerNrShift);
  }
}

// reset the content of a L1GtfeWord
void L1GtfeWord::reset() {
  m_boardId = 0;
  m_recordLength1 = 0;
  m_recordLength = 0;
  m_bxNr = 0;
  m_setupVersion = 0;
  //
  m_activeBoards = 0;
  m_altNrBxBoard = 0;
  m_totalTriggerNr = 0;
}

// pretty print the content of a L1GtfeWord
void L1GtfeWord::print(std::ostream& myCout) const {
  myCout << "\n L1GtfeWord::print \n" << std::endl;

  myCout << "  BoardId:              " << std::hex << " hex: "
         << "      " << std::setw(2) << std::setfill('0') << m_boardId << std::setfill(' ') << std::dec
         << " dec: " << m_boardId << std::endl;

  myCout << "  BX for alternative 1: " << std::hex << " hex: "
         << "       " << std::setw(1) << m_recordLength1 << std::dec << " dec: " << m_recordLength1 << std::endl;

  myCout << "  BX for alternative 0: " << std::hex << " hex: "
         << "       " << std::setw(1) << m_recordLength << std::dec << " dec: " << m_recordLength << std::endl;

  myCout << "  BxNr:                 " << std::hex << " hex: "
         << "     " << std::setw(3) << std::setfill('0') << m_bxNr << std::setfill(' ') << std::dec
         << " dec: " << m_bxNr << std::endl;

  myCout << "  SetupVersion:         " << std::hex << " hex: " << std::setw(8) << std::setfill('0') << m_setupVersion
         << std::setfill(' ') << std::dec << " dec: " << m_setupVersion << std::endl;

  //

  myCout << "  ActiveBoards:         " << std::hex << " hex: "
         << "    " << std::setw(4) << std::setfill('0') << m_activeBoards << std::setfill(' ') << std::dec
         << " dec: " << m_activeBoards << std::endl;

  myCout << "  AltNrBxBoard:         " << std::hex << " hex: "
         << "    " << std::setw(4) << std::setfill('0') << m_altNrBxBoard << std::setfill(' ') << std::dec
         << " dec: " << m_altNrBxBoard << std::endl;

  myCout << "  TotalTriggerNr:       " << std::hex << " hex: " << std::setw(8) << std::setfill('0') << m_totalTriggerNr
         << std::setfill(' ') << std::dec << " dec: " << m_totalTriggerNr << std::endl;
}

// unpack GTFE
// gtfePtr pointer to the beginning of the GTFE block in the raw data

void L1GtfeWord::unpack(const unsigned char* gtfePtr) {
  LogDebug("L1GtfeWord") << "\nUnpacking GTFE block.\n" << std::endl;

  const cms_uint64_t* payload = reinterpret_cast<cms_uint64_t const*>(gtfePtr);

  setBoardId(payload[BoardIdWord]);
  setRecordLength1(payload[RecordLength1Word]);
  setRecordLength(payload[RecordLengthWord]);
  setBxNr(payload[BxNrWord]);
  setSetupVersion(payload[SetupVersionWord]);
  setActiveBoards(payload[ActiveBoardsWord]);
  setAltNrBxBoard(payload[AltNrBxBoardWord]);
  setTotalTriggerNr(payload[TotalTriggerNrWord]);

  if (edm::isDebugEnabled()) {
    for (int iWord = 0; iWord < BlockSize; ++iWord) {
      LogTrace("L1GtfeWord") << std::setw(4) << iWord << "  " << std::hex << std::setfill('0') << std::setw(16)
                             << payload[iWord] << std::dec << std::setfill(' ') << std::endl;
    }
  }
}

// static class members

// block description in the raw GT record

// block size in 64bits words
const int L1GtfeWord::BlockSize = 2;  // 2 x 64bits

const int L1GtfeWord::BoardIdWord = 0;
const int L1GtfeWord::RecordLength1Word = 0;
const int L1GtfeWord::RecordLengthWord = 0;
const int L1GtfeWord::BxNrWord = 0;
const int L1GtfeWord::SetupVersionWord = 0;

const cms_uint64_t L1GtfeWord::BoardIdMask = 0xFF00000000000000ULL;
const cms_uint64_t L1GtfeWord::RecordLength1Mask = 0x00F0000000000000ULL;
const cms_uint64_t L1GtfeWord::RecordLengthMask = 0x000F000000000000ULL;
const cms_uint64_t L1GtfeWord::BxNrMask = 0x00000FFF00000000ULL;
const cms_uint64_t L1GtfeWord::SetupVersionMask = 0x00000000FFFFFFFFULL;

const cms_uint32_t L1GtfeWord::BstFlagMask = 0x0001;

// shifts could be computed from masks...
const int L1GtfeWord::BoardIdShift = 56;
const int L1GtfeWord::RecordLength1Shift = 52;
const int L1GtfeWord::RecordLengthShift = 48;
const int L1GtfeWord::BxNrShift = 32;
const int L1GtfeWord::SetupVersionShift = 0;

//
const int L1GtfeWord::ActiveBoardsWord = 1;
const int L1GtfeWord::AltNrBxBoardWord = 1;
const int L1GtfeWord::TotalTriggerNrWord = 1;

const cms_uint64_t L1GtfeWord::ActiveBoardsMask = 0xFFFF000000000000ULL;
const cms_uint64_t L1GtfeWord::AltNrBxBoardMask = 0x0000FFFF00000000ULL;
const cms_uint64_t L1GtfeWord::TotalTriggerNrMask = 0x00000000FFFFFFFFULL;

const int L1GtfeWord::ActiveBoardsShift = 48;
const int L1GtfeWord::AltNrBxBoardShift = 32;
const int L1GtfeWord::TotalTriggerNrShift = 0;
