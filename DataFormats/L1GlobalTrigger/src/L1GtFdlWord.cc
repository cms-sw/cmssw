/**
 * \class L1GtFdlWord
 *
 *
 * Description: see header file.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"

// system include files
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

// user include files

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors

// empty constructor, all members set to zero;
L1GtFdlWord::L1GtFdlWord() {
  m_boardId = 0;
  m_bxInEvent = 0;
  m_bxNr = 0;
  m_eventNr = 0;

  // technical triggers std::vector<bool>
  m_gtTechnicalTriggerWord.reserve(L1GlobalTriggerReadoutSetup::NumberTechnicalTriggers);
  m_gtTechnicalTriggerWord.assign(L1GlobalTriggerReadoutSetup::NumberTechnicalTriggers, false);

  // decision word  std::vector<bool>
  m_gtDecisionWord.reserve(L1GlobalTriggerReadoutSetup::NumberPhysTriggers);
  m_gtDecisionWord.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers, false);

  // extended decision word  std::vector<bool>
  m_gtDecisionWordExtended.reserve(L1GlobalTriggerReadoutSetup::NumberPhysTriggersExtended);
  m_gtDecisionWordExtended.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggersExtended, false);

  m_physicsDeclared = 0;
  m_gtPrescaleFactorIndexTech = 0;
  m_gtPrescaleFactorIndexAlgo = 0;

  m_noAlgo = 0;

  m_finalOR = 0;

  m_orbitNr = 0;
  m_lumiSegmentNr = 0;
  m_localBxNr = 0;
}

// constructor from unpacked values;
L1GtFdlWord::L1GtFdlWord(cms_uint16_t boardIdValue,
                         int bxInEventValue,
                         cms_uint16_t bxNrValue,
                         cms_uint32_t eventNrValue,
                         const TechnicalTriggerWord& gtTechnicalTriggerWordValue,
                         const DecisionWord& gtDecisionWordValue,
                         const DecisionWordExtended& gtDecisionWordExtendedValue,
                         cms_uint16_t gtPrescaleFactorIndexTechValue,
                         cms_uint16_t gtPrescaleFactorIndexAlgoValue,
                         cms_uint16_t noAlgoValue,
                         cms_uint16_t finalORValue,
                         cms_uint32_t orbitNrValue,
                         cms_uint16_t lumiSegmentNrValue,
                         cms_uint16_t localBxNrValue)
    : m_boardId(boardIdValue),
      m_bxInEvent(bxInEventValue),
      m_bxNr(bxNrValue),
      m_eventNr(eventNrValue),
      m_gtTechnicalTriggerWord(gtTechnicalTriggerWordValue),
      m_gtDecisionWord(gtDecisionWordValue),
      m_gtDecisionWordExtended(gtDecisionWordExtendedValue),
      m_gtPrescaleFactorIndexTech(gtPrescaleFactorIndexTechValue),
      m_gtPrescaleFactorIndexAlgo(gtPrescaleFactorIndexAlgoValue),
      m_noAlgo(noAlgoValue),
      m_finalOR(finalORValue),
      m_orbitNr(orbitNrValue),
      m_lumiSegmentNr(lumiSegmentNrValue),
      m_localBxNr(localBxNrValue)

{
  m_physicsDeclared = 0;

  // the rest done in initialization list
}

// destructor
L1GtFdlWord::~L1GtFdlWord() {
  // empty now
}

// equal operator
bool L1GtFdlWord::operator==(const L1GtFdlWord& result) const {
  if (m_boardId != result.m_boardId) {
    return false;
  }

  if (m_bxInEvent != result.m_bxInEvent) {
    return false;
  }

  if (m_bxNr != result.m_bxNr) {
    return false;
  }
  if (m_eventNr != result.m_eventNr) {
    return false;
  }

  if (m_gtTechnicalTriggerWord != result.m_gtTechnicalTriggerWord) {
    return false;
  }

  if (m_gtDecisionWord != result.m_gtDecisionWord) {
    return false;
  }

  if (m_gtDecisionWordExtended != result.m_gtDecisionWordExtended) {
    return false;
  }

  if (m_physicsDeclared != result.m_physicsDeclared) {
    return false;
  }

  if (m_gtPrescaleFactorIndexTech != result.m_gtPrescaleFactorIndexTech) {
    return false;
  }

  if (m_gtPrescaleFactorIndexAlgo != result.m_gtPrescaleFactorIndexAlgo) {
    return false;
  }

  if (m_noAlgo != result.m_noAlgo) {
    return false;
  }

  if (m_finalOR != result.m_finalOR) {
    return false;
  }

  if (m_orbitNr != result.m_orbitNr) {
    return false;
  }

  if (m_lumiSegmentNr != result.m_lumiSegmentNr) {
    return false;
  }

  if (m_localBxNr != result.m_localBxNr) {
    return false;
  }

  // all members identical
  return true;
}

// unequal operator
bool L1GtFdlWord::operator!=(const L1GtFdlWord& result) const { return !(result == *this); }

// methods

// set the BoardId value from a 64-bits word
void L1GtFdlWord::setBoardId(const cms_uint64_t& word64) { m_boardId = (word64 & BoardIdMask) >> BoardIdShift; }

// set the BoardId value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setBoardIdWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == BoardIdWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_boardId) << BoardIdShift);
  }
}

// set the BxInEvent value from a 64-bits word
void L1GtFdlWord::setBxInEvent(const cms_uint64_t& word64) {
  int baseValue = 16;  // using hexadecimal values;
  int hexBxInEvent = (word64 & BxInEventMask) >> BxInEventShift;
  m_bxInEvent = (hexBxInEvent + baseValue / 2) % baseValue - baseValue / 2;
}

// set the BxInEvent value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setBxInEventWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == BxInEventWord) {
    int baseValue = 16;  // using hexadecimal values;
    int hexBxInEvent = (m_bxInEvent + baseValue) % baseValue;
    word64 = word64 | (static_cast<cms_uint64_t>(hexBxInEvent) << BxInEventShift);
  }
}

// set the BxNr value from a 64-bits word
void L1GtFdlWord::setBxNr(const cms_uint64_t& word64) { m_bxNr = (word64 & BxNrMask) >> BxNrShift; }

// set the BxNr value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setBxNrWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == BxNrWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_bxNr) << BxNrShift);
  }
}

// set the EventNr value from a 64-bits word
void L1GtFdlWord::setEventNr(const cms_uint64_t& word64) { m_eventNr = (word64 & EventNrMask) >> EventNrShift; }

// set the EventNr value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setEventNrWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == EventNrWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_eventNr) << EventNrShift);
  }
}

// print GT technical trigger word in bitset style
//    depend on the type of TechnicalTriggerWord
//    this version: <vector<bool>
void L1GtFdlWord::printGtTechnicalTriggerWord(std::ostream& myCout) const {
  myCout << "  Technical triggers (bitset style):    \n  ";

  int sizeW64 = 64;  // 64 bits words
  int iBit = 0;

  for (std::vector<bool>::const_reverse_iterator ritBit = m_gtTechnicalTriggerWord.rbegin();
       ritBit != m_gtTechnicalTriggerWord.rend();
       ++ritBit) {
    myCout << (*ritBit ? '1' : '0');

    if ((((iBit + 1) % 16) == (sizeW64 % 16)) && (iBit != 63)) {
      myCout << " ";
    }

    iBit++;
  }
}

// set the GtTechnicalTriggerWord value from a 64-bits word
void L1GtFdlWord::setGtTechnicalTriggerWord(const cms_uint64_t& word64) {
  int word64Size = sizeof(word64) * 8;
  cms_uint64_t wordTT = (word64 & GtTechnicalTriggerWordMask) >> GtTechnicalTriggerWordShift;

  cms_uint64_t one64 = 1ULL;
  for (int iBit = 0; iBit < word64Size; ++iBit) {
    m_gtTechnicalTriggerWord.at(iBit) = wordTT & (one64 << iBit);
  }
}

// set the GtTechnicalTriggerWord value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setGtTechnicalTriggerWordWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == GtTechnicalTriggerWordWord) {
    int word64Size = sizeof(word64) * 8;
    cms_uint64_t wordTT = 0x0000000000000000ULL;

    int iBit = 0;
    cms_uint64_t iDecision = 0ULL;

    for (std::vector<bool>::const_iterator itBit = m_gtTechnicalTriggerWord.begin();
         itBit != m_gtTechnicalTriggerWord.end();
         ++itBit) {
      iDecision = static_cast<cms_uint64_t>(*itBit);  //(*itBit ? 1 : 0);
      wordTT = wordTT | (iDecision << iBit);

      iBit++;
      if (iBit >= word64Size) {
        break;
      }
    }

    word64 = word64 | (wordTT << GtTechnicalTriggerWordShift);
  }
}

// print GT decision word in bitset style
//    depend on the type of DecisionWord
//    this version: <vector<bool>
void L1GtFdlWord::printGtDecisionWord(std::ostream& myCout) const {
  // decision word (in 64bits words)
  int sizeW64 = 64;  // 64 bits words

  int iBit = 0;
  int nrDecWord = m_gtDecisionWord.size() / sizeW64;

  std::ostringstream stream64;

  std::vector<std::string> decWord;
  decWord.reserve(nrDecWord);

  for (std::vector<bool>::const_reverse_iterator ritBit = m_gtDecisionWord.rbegin(); ritBit != m_gtDecisionWord.rend();
       ++ritBit) {
    stream64 << (*ritBit ? '1' : '0');

    if ((((iBit + 1) % 16) == (sizeW64 % 16))) {
      stream64 << " ";
    }

    if (((iBit + 1) % sizeW64) == 0) {
      std::string iW = stream64.str();
      stream64.str("");

      decWord.push_back(iW);
    }

    iBit++;
  }

  int iWord = 0;

  for (std::vector<std::string>::reverse_iterator ritWord = decWord.rbegin(); ritWord != decWord.rend(); ++ritWord) {
    myCout << std::endl;
    myCout << "  DecisionWord (bitset style): bits " << iWord * sizeW64 + sizeW64 - 1 << " : " << iWord * sizeW64
           << "\n  ";
    myCout << *ritWord;

    iWord++;
  }
}

// set the GtDecisionWord value from a 64-bits word
// WordA: bits 0 - 63
void L1GtFdlWord::setGtDecisionWordA(const cms_uint64_t& word64) {
  int word64Size = sizeof(word64) * 8;  // well, it should be 64, if not...  :-)
  cms_uint64_t wordA = (word64 & GtDecisionWordAMask) >> GtDecisionWordAShift;

  cms_uint64_t one64 = 1ULL;

  for (int iBit = 0; iBit < word64Size; ++iBit) {
    m_gtDecisionWord.at(iBit) = wordA & (one64 << iBit);
  }
}

// set the GtDecisionWord value from a 64-bits word
// WordB: bits 64 - 127
void L1GtFdlWord::setGtDecisionWordB(const cms_uint64_t& word64) {
  int word64Size = sizeof(word64) * 8;
  cms_uint64_t wordB = (word64 & GtDecisionWordBMask) >> GtDecisionWordBShift;

  cms_uint64_t one64 = 1ULL;

  for (int iBit = 0; iBit < word64Size; ++iBit) {
    m_gtDecisionWord.at(iBit + word64Size) = wordB & (one64 << iBit);
  }
}

// set the GtDecisionWord value in a 64-bits word, having the index iWord
// in the GTFE raw record
// WordA: bits 0 - 63

// a bit forced: assumes wordSize64 = 64, but also take word shift
void L1GtFdlWord::setGtDecisionWordAWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == GtDecisionWordAWord) {
    int word64Size = sizeof(word64) * 8;
    cms_uint64_t wordA = 0x0000000000000000ULL;

    int iBit = 0;
    cms_uint64_t iDecision = 0ULL;

    for (std::vector<bool>::const_iterator itBit = m_gtDecisionWord.begin(); itBit != m_gtDecisionWord.end(); ++itBit) {
      iDecision = static_cast<cms_uint64_t>(*itBit);  //(*itBit ? 1 : 0);
      wordA = wordA | (iDecision << iBit);

      iBit++;
      if (iBit >= word64Size) {
        break;
      }
    }

    word64 = word64 | (wordA << GtDecisionWordAShift);
  }
}

// set the GtDecisionWord value in a 64-bits word, having the index iWord
// in the GTFE raw record
// WordB: bits 64 - 127

// a bit forced: assumes wordSize64 = 64, but also take word shift
void L1GtFdlWord::setGtDecisionWordBWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == GtDecisionWordBWord) {
    int word64Size = sizeof(word64) * 8;
    cms_uint64_t wordB = 0x0000000000000000ULL;

    int iBit = 0;
    cms_uint64_t iDecision = 0ULL;

    for (std::vector<bool>::const_iterator itBit = m_gtDecisionWord.begin(); itBit != m_gtDecisionWord.end(); ++itBit) {
      if (iBit >= word64Size) {
        // skip first word64Size bits, they go in wordA
        iDecision = static_cast<cms_uint64_t>(*itBit);  //(*itBit ? 1 : 0);
        wordB = wordB | (iDecision << (iBit - word64Size));
      }

      iBit++;
    }

    word64 = word64 | (wordB << GtDecisionWordBShift);
  }
}

// print GT decision word extended in bitset style
//    depend on the type of DecisionWord
//    this version: <vector<bool>
void L1GtFdlWord::printGtDecisionWordExtended(std::ostream& myCout) const {
  myCout << "  DecisionWordExtended (bitset style):    \n  ";

  int sizeW64 = 64;  // 64 bits words
  int iBit = 0;

  for (std::vector<bool>::const_reverse_iterator ritBit = m_gtDecisionWordExtended.rbegin();
       ritBit != m_gtDecisionWordExtended.rend();
       ++ritBit) {
    myCout << (*ritBit ? '1' : '0');

    if ((((iBit + 1) % 16) == (sizeW64 % 16)) && (iBit != 63)) {
      myCout << " ";
    }

    iBit++;
  }
}

// set the GtDecisionWordExtended value from a 64-bits word
void L1GtFdlWord::setGtDecisionWordExtended(const cms_uint64_t& word64) {
  int word64Size = sizeof(word64) * 8;
  cms_uint64_t wordE = (word64 & GtDecisionWordExtendedMask) >> GtDecisionWordExtendedShift;

  cms_uint64_t one64 = 1ULL;

  for (int iBit = 0; iBit < word64Size; ++iBit) {
    m_gtDecisionWordExtended.at(iBit) = wordE & (one64 << iBit);
  }
}

// set the GtDecisionWordExtended value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setGtDecisionWordExtendedWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == GtDecisionWordExtendedWord) {
    int word64Size = sizeof(word64) * 8;
    cms_uint64_t wordE = 0x0000000000000000ULL;

    int iBit = 0;
    cms_uint64_t iDecision = 0ULL;

    for (std::vector<bool>::const_iterator itBit = m_gtDecisionWordExtended.begin();
         itBit != m_gtDecisionWordExtended.end();
         ++itBit) {
      iDecision = static_cast<cms_uint64_t>(*itBit);  //(*itBit ? 1 : 0);
      wordE = wordE | (iDecision << iBit);

      iBit++;
      if (iBit >= word64Size) {
        break;
      }
    }

    word64 = word64 | (wordE << GtDecisionWordExtendedShift);
  }
}

// set the "physics declared" bit value from a 64-bits word
void L1GtFdlWord::setPhysicsDeclared(const cms_uint64_t& word64) {
  m_physicsDeclared = (word64 & PhysicsDeclaredMask) >> PhysicsDeclaredShift;
}

// set the "physics declared" bit value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setPhysicsDeclaredWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == PhysicsDeclaredWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_physicsDeclared) << PhysicsDeclaredShift);
  }
}

// set the GtPrescaleFactorIndexTech from a 64-bits word
void L1GtFdlWord::setGtPrescaleFactorIndexTech(const cms_uint64_t& word64) {
  m_gtPrescaleFactorIndexTech = (word64 & GtPrescaleFactorIndexTechMask) >> GtPrescaleFactorIndexTechShift;
}

// set the GtPrescaleFactorIndexTech bits in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setGtPrescaleFactorIndexTechWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == GtPrescaleFactorIndexTechWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_gtPrescaleFactorIndexTech) << GtPrescaleFactorIndexTechShift);
  }
}

// set the GtPrescaleFactorIndexAlgo from a 64-bits word
void L1GtFdlWord::setGtPrescaleFactorIndexAlgo(const cms_uint64_t& word64) {
  m_gtPrescaleFactorIndexAlgo = (word64 & GtPrescaleFactorIndexAlgoMask) >> GtPrescaleFactorIndexAlgoShift;
}

// set the GtPrescaleFactorIndexAlgo bits in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setGtPrescaleFactorIndexAlgoWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == GtPrescaleFactorIndexAlgoWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_gtPrescaleFactorIndexAlgo) << GtPrescaleFactorIndexAlgoShift);
  }
}

// set the NoAlgo value from a 64-bits word
void L1GtFdlWord::setNoAlgo(const cms_uint64_t& word64) { m_noAlgo = (word64 & NoAlgoMask) >> NoAlgoShift; }

// set the NoAlgo value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setNoAlgoWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == NoAlgoWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_noAlgo) << NoAlgoShift);
  }
}

// set the FinalOR value from a 64-bits word
void L1GtFdlWord::setFinalOR(const cms_uint64_t& word64) { m_finalOR = (word64 & FinalORMask) >> FinalORShift; }

// set the FinalOR value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setFinalORWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == FinalORWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_finalOR) << FinalORShift);
  }
}

// set the orbit number bits from a 64-bits word
void L1GtFdlWord::setOrbitNr(const cms_uint64_t& word64) { m_orbitNr = (word64 & OrbitNrMask) >> OrbitNrShift; }

// set the orbit number bits in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setOrbitNrWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == OrbitNrWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_orbitNr) << OrbitNrShift);
  }
}

// set the luminosity segment number bits from a 64-bits word
void L1GtFdlWord::setLumiSegmentNr(const cms_uint64_t& word64) {
  m_lumiSegmentNr = (word64 & LumiSegmentNrMask) >> LumiSegmentNrShift;
}

// set the luminosity segment number bits in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setLumiSegmentNrWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == LumiSegmentNrWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_lumiSegmentNr) << LumiSegmentNrShift);
  }
}

// set the LocalBxNr value from a 64-bits word
void L1GtFdlWord::setLocalBxNr(const cms_uint64_t& word64) { m_localBxNr = (word64 & LocalBxNrMask) >> LocalBxNrShift; }

// set the LocalBxNr value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setLocalBxNrWord64(cms_uint64_t& word64, const int iWord) {
  if (iWord == LocalBxNrWord) {
    word64 = word64 | (static_cast<cms_uint64_t>(m_localBxNr) << LocalBxNrShift);
  }
}

// reset the content of a L1GtFdlWord
void L1GtFdlWord::reset() {
  m_boardId = 0;
  m_bxInEvent = 0;
  m_bxNr = 0;
  m_eventNr = 0;

  // technical triggers std::vector<bool>
  m_gtTechnicalTriggerWord.assign(L1GlobalTriggerReadoutSetup::NumberTechnicalTriggers, false);

  // decision word  std::vector<bool>
  m_gtDecisionWord.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers, false);

  // extended decision word  std::vector<bool>
  m_gtDecisionWordExtended.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggersExtended, false);

  m_physicsDeclared = 0;
  m_gtPrescaleFactorIndexTech = 0;
  m_gtPrescaleFactorIndexAlgo = 0;

  m_noAlgo = 0;
  m_finalOR = 0;

  m_orbitNr = 0;
  m_lumiSegmentNr = 0;
  m_localBxNr = 0;
}

// pretty print the content of a L1GtFdlWord
void L1GtFdlWord::print(std::ostream& myCout) const {
  myCout << "\n L1GtFdlWord::print \n" << std::endl;

  int iWord = 0;

  myCout << "\n Word " << iWord << std::endl;

  myCout << "  Board Id:         " << std::hex << " hex: "
         << "    " << std::setw(4) << std::setfill('0') << m_boardId << std::setfill(' ') << std::dec
         << " dec: " << m_boardId << std::endl;
  //

  int baseValue = 16;  // using hexadecimal values;
  int hexBxInEvent = (m_bxInEvent + baseValue) % baseValue;

  myCout << "  BxInEvent:        " << std::hex << " hex: "
         << "       " << std::setw(1) << hexBxInEvent << std::dec << " dec: " << m_bxInEvent << std::endl;

  myCout << "  BxNr:             " << std::hex << " hex: "
         << "     " << std::setw(3) << std::setfill('0') << m_bxNr << std::setfill(' ') << std::dec
         << " dec: " << m_bxNr << std::endl;

  myCout << "  EventNr:          " << std::hex << " hex: "
         << "  " << std::setw(6) << std::setfill('0') << m_eventNr << std::setfill(' ') << std::dec
         << " dec: " << m_eventNr << std::endl;

  // technical triggers

  iWord++;
  myCout << "\n Word " << iWord << std::endl;

  printGtTechnicalTriggerWord(myCout);
  myCout << std::endl;

  // physics triggers (2 words!)

  iWord++;
  myCout << "\n Word " << iWord;
  iWord++;
  myCout << " and word " << iWord;

  printGtDecisionWord(myCout);
  myCout << std::endl;

  // decision word extended (64 bits)

  iWord++;
  myCout << "\n Word " << iWord << std::endl;

  printGtDecisionWordExtended(myCout);
  myCout << std::endl;

  //
  iWord++;
  myCout << "\n Word " << iWord << std::endl;

  myCout << "  PhysicsDeclared:           " << std::hex << " hex: "
         << "    " << std::setw(4) << std::setfill('0') << m_physicsDeclared << std::setfill(' ') << std::dec
         << " dec: " << m_physicsDeclared << std::endl;

  myCout << "  GtPrescaleFactorIndexTech: " << std::hex << " hex: "
         << "    " << std::setw(4) << std::setfill('0') << m_gtPrescaleFactorIndexTech << std::setfill(' ') << std::dec
         << " dec: " << m_gtPrescaleFactorIndexTech << std::endl;

  myCout << "  GtPrescaleFactorIndexAlgo: " << std::hex << " hex: "
         << "    " << std::setw(4) << std::setfill('0') << m_gtPrescaleFactorIndexAlgo << std::setfill(' ') << std::dec
         << " dec: " << m_gtPrescaleFactorIndexAlgo << std::endl;

  myCout << "  NoAlgo:                    " << std::hex << " hex: "
         << "       " << std::setw(1) << std::setfill('0') << m_noAlgo << std::setfill(' ') << std::dec
         << " dec: " << m_noAlgo << std::endl;

  myCout << "  FinalOR:                   " << std::hex << " hex: "
         << "      " << std::setw(2) << std::setfill('0') << m_finalOR << std::setfill(' ') << std::dec
         << " dec: " << m_finalOR << std::endl;

  iWord++;
  myCout << "\n Word " << iWord << std::endl;

  myCout << "  OrbitNr:          " << std::hex << " hex: "
         << "" << std::setw(8) << std::setfill('0') << m_orbitNr << std::setfill(' ') << std::dec
         << " dec: " << m_orbitNr << std::endl;

  myCout << "  LumiSegmentNr:    " << std::hex << " hex: "
         << "    " << std::setw(4) << std::setfill('0') << m_lumiSegmentNr << std::setfill(' ') << std::dec
         << " dec: " << m_lumiSegmentNr << std::endl;

  myCout << "  LocalBxNr:        " << std::hex << " hex: "
         << "     " << std::setw(3) << std::setfill('0') << m_localBxNr << std::setfill(' ') << std::dec
         << " dec: " << m_localBxNr << std::endl;
}

// unpack FDL
// fdlPtr pointer to the beginning of the FDL block in the raw data
void L1GtFdlWord::unpack(const unsigned char* fdlPtr) {
  LogDebug("L1GtFdlWord") << "\nUnpacking FDL block.\n" << std::endl;

  const cms_uint64_t* payload = reinterpret_cast<cms_uint64_t const*>(fdlPtr);

  setBoardId(payload[BoardIdWord]);
  setBxInEvent(payload[BxInEventWord]);
  setBxNr(payload[BxNrWord]);
  setEventNr(payload[EventNrWord]);

  setGtTechnicalTriggerWord(payload[GtTechnicalTriggerWordWord]);

  setGtDecisionWordA(payload[GtDecisionWordAWord]);

  setGtDecisionWordB(payload[GtDecisionWordBWord]);

  setGtDecisionWordExtended(payload[GtDecisionWordExtendedWord]);

  setPhysicsDeclared(payload[PhysicsDeclaredWord]);
  setGtPrescaleFactorIndexTech(payload[GtPrescaleFactorIndexTechWord]);
  setGtPrescaleFactorIndexAlgo(payload[GtPrescaleFactorIndexAlgoWord]);
  setNoAlgo(payload[NoAlgoWord]);
  setFinalOR(payload[FinalORWord]);

  setOrbitNr(payload[OrbitNrWord]);
  setLumiSegmentNr(payload[LumiSegmentNrWord]);
  setLocalBxNr(payload[LocalBxNrWord]);

  if (edm::isDebugEnabled()) {
    for (int iWord = 0; iWord < BlockSize; ++iWord) {
      LogTrace("L1GtFdlWord") << std::setw(4) << iWord << "  " << std::hex << std::setfill('0') << std::setw(16)
                              << payload[iWord] << std::dec << std::setfill(' ') << std::endl;
    }
  }
}

// static class members

// block description in the raw GT record

// block size in 64bits words (BlockSize * 64 bits)
const int L1GtFdlWord::BlockSize = 7;

// word 0

// index of the word in the FDL block containig the variable
const int L1GtFdlWord::BoardIdWord = 0;
const int L1GtFdlWord::BxInEventWord = 0;
const int L1GtFdlWord::BxNrWord = 0;
const int L1GtFdlWord::EventNrWord = 0;

// mask to get the 64-bit-value from the corresponding word in the FDL block
const cms_uint64_t L1GtFdlWord::BoardIdMask = 0xFFFF000000000000ULL;
const cms_uint64_t L1GtFdlWord::BxInEventMask = 0x0000F00000000000ULL;
const cms_uint64_t L1GtFdlWord::BxNrMask = 0x00000FFF00000000ULL;
const cms_uint64_t L1GtFdlWord::EventNrMask = 0x0000000000FFFFFFULL;

// shift to the right to get the value from the "64-bit-value"
const int L1GtFdlWord::BoardIdShift = 48;
const int L1GtFdlWord::BxInEventShift = 44;
const int L1GtFdlWord::BxNrShift = 32;
const int L1GtFdlWord::EventNrShift = 0;

// word 1

const int L1GtFdlWord::GtTechnicalTriggerWordWord = 1;
const cms_uint64_t L1GtFdlWord::GtTechnicalTriggerWordMask = 0xFFFFFFFFFFFFFFFFULL;
const int L1GtFdlWord::GtTechnicalTriggerWordShift = 0;

// word 2 - WordA: bits 0-63

const int L1GtFdlWord::GtDecisionWordAWord = 2;
const cms_uint64_t L1GtFdlWord::GtDecisionWordAMask = 0xFFFFFFFFFFFFFFFFULL;
const int L1GtFdlWord::GtDecisionWordAShift = 0;

// word 3 - WordB: bits 64-128

const int L1GtFdlWord::GtDecisionWordBWord = 3;
const cms_uint64_t L1GtFdlWord::GtDecisionWordBMask = 0xFFFFFFFFFFFFFFFFULL;
const int L1GtFdlWord::GtDecisionWordBShift = 0;

// word 4
const int L1GtFdlWord::GtDecisionWordExtendedWord = 4;
const cms_uint64_t L1GtFdlWord::GtDecisionWordExtendedMask = 0xFFFFFFFFFFFFFFFFULL;
const int L1GtFdlWord::GtDecisionWordExtendedShift = 0;

// word 5
const int L1GtFdlWord::PhysicsDeclaredWord = 5;
const int L1GtFdlWord::GtPrescaleFactorIndexTechWord = 5;
const int L1GtFdlWord::GtPrescaleFactorIndexAlgoWord = 5;
const int L1GtFdlWord::NoAlgoWord = 5;
const int L1GtFdlWord::FinalORWord = 5;

const cms_uint64_t L1GtFdlWord::PhysicsDeclaredMask = 0x8000000000000000ULL;
const cms_uint64_t L1GtFdlWord::GtPrescaleFactorIndexTechMask = 0x00FF000000000000ULL;
const cms_uint64_t L1GtFdlWord::GtPrescaleFactorIndexAlgoMask = 0x000000FF00000000ULL;
const cms_uint64_t L1GtFdlWord::NoAlgoMask = 0x0000000000000100ULL;
const cms_uint64_t L1GtFdlWord::FinalORMask = 0x00000000000000FFULL;

const int L1GtFdlWord::PhysicsDeclaredShift = 63;
const int L1GtFdlWord::GtPrescaleFactorIndexTechShift = 48;
const int L1GtFdlWord::GtPrescaleFactorIndexAlgoShift = 32;
const int L1GtFdlWord::NoAlgoShift = 8;
const int L1GtFdlWord::FinalORShift = 0;

// word 6
const int L1GtFdlWord::OrbitNrWord = 6;
const int L1GtFdlWord::LumiSegmentNrWord = 6;
const int L1GtFdlWord::LocalBxNrWord = 6;

const cms_uint64_t L1GtFdlWord::OrbitNrMask = 0xFFFFFFFF00000000ULL;
const cms_uint64_t L1GtFdlWord::LumiSegmentNrMask = 0x00000000FFFF0000ULL;
const cms_uint64_t L1GtFdlWord::LocalBxNrMask = 0x0000000000000FFFULL;

const int L1GtFdlWord::OrbitNrShift = 32;
const int L1GtFdlWord::LumiSegmentNrShift = 16;
const int L1GtFdlWord::LocalBxNrShift = 0;
