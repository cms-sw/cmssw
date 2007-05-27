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
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"

// system include files
#include <iostream>
#include <iomanip>
#include <vector>
#include <boost/cstdint.hpp>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "FWCore/Utilities/interface/EDMException.h"


// constructors

// empty constructor, all members set to zero;
L1GtFdlWord::L1GtFdlWord()
{

    m_boardId = 0;
    m_bxInEvent = 0;
    m_bxNr = 0;
    m_eventNr = 0;

    // technical triggers std::vector<bool>
    m_gtTechnicalTriggerWord.reserve(
        L1GlobalTriggerReadoutSetup::NumberTechnicalTriggers);
    m_gtTechnicalTriggerWord.assign(
        L1GlobalTriggerReadoutSetup::NumberTechnicalTriggers, false);

    // decision word  std::vector<bool>
    m_gtDecisionWord.reserve(
        L1GlobalTriggerReadoutSetup::NumberPhysTriggers);
    m_gtDecisionWord.assign(
        L1GlobalTriggerReadoutSetup::NumberPhysTriggers, false);

    // extended decision word  std::vector<bool>
    m_gtDecisionWordExtended.reserve(
        L1GlobalTriggerReadoutSetup::NumberPhysTriggersExtended);
    m_gtDecisionWordExtended.assign(
        L1GlobalTriggerReadoutSetup::NumberPhysTriggersExtended, false);

    m_noAlgo = 0;

    m_finalOR = 0;
    m_localBxNr = 0;

}

// constructor from unpacked values;
L1GtFdlWord::L1GtFdlWord(
    boost::uint16_t boardIdValue,
    int bxInEventValue,
    boost::uint16_t bxNrValue,
    boost::uint32_t eventNrValue,
    TechnicalTriggerWord gtTechnicalTriggerWordValue,
    DecisionWord gtDecisionWordValue,
    DecisionWordExtended gtDecisionWordExtendedValue,
    boost::uint16_t noAlgoValue,
    boost::uint16_t finalORValue,
    boost::uint16_t localBxNrValue
)
{

    m_boardId = boardIdValue;
    m_bxInEvent = bxInEventValue;
    m_bxNr = bxNrValue;
    m_eventNr = eventNrValue;
    m_gtTechnicalTriggerWord = gtTechnicalTriggerWordValue;
    m_gtDecisionWord = gtDecisionWordValue;
    m_gtDecisionWordExtended = gtDecisionWordExtendedValue;
    m_noAlgo = noAlgoValue;
    m_finalOR = finalORValue;
    m_localBxNr = localBxNrValue;

}

// destructor
L1GtFdlWord::~L1GtFdlWord()
{

    // empty now
}

// equal operator
bool L1GtFdlWord::operator==(const L1GtFdlWord& result) const
{

    if (m_boardId   != result.m_boardId) {
        return false;
    }

    if (m_bxInEvent != result.m_bxInEvent) {
        return false;
    }

    if (m_bxNr      != result.m_bxNr) {
        return false;
    }
    if (m_eventNr   != result.m_eventNr) {
        return false;
    }

    if (m_gtTechnicalTriggerWord != result.m_gtTechnicalTriggerWord) {
        return false;
    }

    if (m_gtDecisionWord         != result.m_gtDecisionWord) {
        return false;
    }

    if (m_gtDecisionWordExtended != result.m_gtDecisionWordExtended) {
        return false;
    }

    if (m_noAlgo                != result.m_noAlgo) {
        return false;
    }

    if (m_finalOR               != result.m_finalOR) {
        return false;
    }

    if (m_localBxNr != result.m_localBxNr) {
        return false;
    }

    // all members identical
    return true;

}

// unequal operator
bool L1GtFdlWord::operator!=(const L1GtFdlWord& result) const
{

    return !( result == *this);

}


// methods

// set the BoardId value from a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setBoardId(const boost::uint64_t& word64, int iWord)
{
    if (iWord == BoardIdWord) {
        m_boardId = (word64 & BoardIdMask) >> BoardIdShift;
    }

}

// set the BoardId value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setBoardIdWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == BoardIdWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_boardId) << BoardIdShift);
    }

}


// set the BxInEvent value from a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setBxInEvent(const boost::uint64_t& word64, int iWord)
{
    if (iWord == BxInEventWord) {
        int baseValue = 16; // using hexadecimal values;
        int hexBxInEvent = (word64 & BxInEventMask) >> BxInEventShift;
        m_bxInEvent = (hexBxInEvent + baseValue/2)%baseValue - baseValue/2;
    }

}

// set the BxInEvent value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setBxInEventWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == BxInEventWord) {
        int baseValue = 16; // using hexadecimal values;
        int hexBxInEvent = (m_bxInEvent + baseValue)%baseValue;
        word64 = word64 | (static_cast<boost::uint64_t> (hexBxInEvent)
                           << BxInEventShift);
    }

}


// set the BxNr value from a 64-bits word, having the index iWord in the GTFE raw record
void L1GtFdlWord::setBxNr(const boost::uint64_t& word64, int iWord)
{
    if (iWord == BxNrWord) {
        m_bxNr = (word64 & BxNrMask) >> BxNrShift;
    }

}

// set the BxNr value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setBxNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == BxNrWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_bxNr) << BxNrShift);
    }

}


// set the EventNr value from a 64-bits word, having the index iWord in the GTFE raw record
void L1GtFdlWord::setEventNr(const boost::uint64_t& word64, int iWord)
{
    if (iWord == EventNrWord) {
        m_eventNr = (word64 & EventNrMask) >> EventNrShift;
    }

}

// set the EventNr value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setEventNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == EventNrWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_eventNr) << EventNrShift);
    }

}


// print GT technical trigger word in bitset style
//    depend on the type of TechnicalTriggerWord
//    this version: <vector<bool>
void L1GtFdlWord::printGtTechnicalTriggerWord(std::ostream& myCout) const
{

    for (std::vector<bool>::const_reverse_iterator ritBit = m_gtTechnicalTriggerWord.rbegin();
            ritBit != m_gtTechnicalTriggerWord.rend(); ++ritBit) {

        myCout << (*ritBit ? '1' : '0');

    }


}

// set the GtTechnicalTriggerWord value from a 64-bits word,
// having the index iWord in the GTFE raw record
void L1GtFdlWord::setGtTechnicalTriggerWord(const boost::uint64_t& word64, int iWord)
{
    if (iWord == GtTechnicalTriggerWordWord) {

        int word64Size = sizeof(word64)*8;
        boost::uint64_t wordTT = (word64 & GtTechnicalTriggerWordMask)
                                 >> GtTechnicalTriggerWordShift;

        boost::uint64_t one64 = 1ULL;
        for (int iBit = 0; iBit < word64Size; ++iBit) {
            m_gtTechnicalTriggerWord.at(iBit) = wordTT & (one64 << iBit);
        }

    }

}

// set the GtTechnicalTriggerWord value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setGtTechnicalTriggerWordWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == GtTechnicalTriggerWordWord) {

        int word64Size = sizeof(word64)*8;
        boost::uint64_t wordTT = 0x0000000000000000ULL;

        int iBit = 0;
        boost::uint64_t iDecision = 0ULL;

        for (std::vector<bool>::const_iterator itBit = m_gtTechnicalTriggerWord.begin();
                itBit != m_gtTechnicalTriggerWord.end(); ++itBit) {

            iDecision = static_cast<boost::uint64_t> (*itBit);//(*itBit ? 1 : 0);
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
void L1GtFdlWord::printGtDecisionWord(std::ostream& myCout) const
{

    for (std::vector<bool>::const_reverse_iterator ritBit = m_gtDecisionWord.rbegin();
            ritBit != m_gtDecisionWord.rend(); ++ritBit) {

        myCout << (*ritBit ? '1' : '0');

    }


}

// set the GtDecisionWord value from a 64-bits word,
// having the index iWord in the GTFE raw record
// WordA: bits 0 - 63
void L1GtFdlWord::setGtDecisionWordA(const boost::uint64_t& word64, int iWord)
{
    if (iWord == GtDecisionWordAWord) {

        int word64Size = sizeof(word64)*8; // well, it should be 64, if not...  :-)
        boost::uint64_t wordA = (word64 & GtDecisionWordAMask) >> GtDecisionWordAShift;

        boost::uint64_t one64 = 1ULL;

        for (int iBit = 0; iBit < word64Size; ++iBit) {
            m_gtDecisionWord.at(iBit) = wordA & (one64 << iBit);
        }

    }

}

// set the GtDecisionWord value from a 64-bits word,
// having the index iWord in the GTFE raw record
// WordB: bits 64 - 127
void L1GtFdlWord::setGtDecisionWordB(const boost::uint64_t& word64, int iWord)
{
    if (iWord == GtDecisionWordBWord) {

        int word64Size = sizeof(word64)*8;
        boost::uint64_t wordB = (word64 & GtDecisionWordBMask) >> GtDecisionWordBShift;

        boost::uint64_t one64 = 1ULL;

        for (int iBit = 0; iBit < word64Size; ++iBit) {
            m_gtDecisionWord.at(iBit + word64Size) = wordB & (one64 << iBit);

        }

    }

}


// set the GtDecisionWord value in a 64-bits word, having the index iWord
// in the GTFE raw record
// WordA: bits 0 - 63

// a bit forced: assumes wordSize64 = 64, but also take word shift
void L1GtFdlWord::setGtDecisionWordAWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == GtDecisionWordAWord) {

        int word64Size = sizeof(word64)*8;
        boost::uint64_t wordA = 0x0000000000000000ULL;

        int iBit = 0;
        boost::uint64_t iDecision = 0ULL;

        for (std::vector<bool>::const_iterator itBit = m_gtDecisionWord.begin();
                itBit != m_gtDecisionWord.end(); ++itBit) {

            iDecision = static_cast<boost::uint64_t> (*itBit);//(*itBit ? 1 : 0);
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
void L1GtFdlWord::setGtDecisionWordBWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == GtDecisionWordBWord) {

        int word64Size = sizeof(word64)*8;
        boost::uint64_t wordB = 0x0000000000000000ULL;

        int iBit = 0;
        boost::uint64_t iDecision = 0ULL;

        for (std::vector<bool>::const_iterator itBit = m_gtDecisionWord.begin();
                itBit != m_gtDecisionWord.end(); ++itBit) {

            if (iBit >= word64Size) {
                // skip first word64Size bits, they go in wordA
                iDecision = static_cast<boost::uint64_t> (*itBit);//(*itBit ? 1 : 0);
                wordB = wordB | ( iDecision << (iBit - word64Size) );
            }

            iBit++;

        }

        word64 = word64 | (wordB << GtDecisionWordBShift);
    }

}



// set the GtDecisionWordExtended value from a 64-bits word,
// having the index iWord in the GTFE raw record
void L1GtFdlWord::setGtDecisionWordExtended(const boost::uint64_t& word64, int iWord)
{
    if (iWord == GtDecisionWordExtendedWord) {

        int word64Size = sizeof(word64)*8;
        boost::uint64_t wordE = (word64 & GtDecisionWordExtendedMask)
                                >> GtDecisionWordExtendedShift;

        boost::uint64_t one64 = 1ULL;

        for (int iBit = 0; iBit < word64Size; ++iBit) {
            m_gtDecisionWordExtended.at(iBit) = wordE & (one64 << iBit);
        }


    }

}

// set the GtDecisionWordExtended value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setGtDecisionWordExtendedWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == GtDecisionWordExtendedWord) {

        int word64Size = sizeof(word64)*8;
        boost::uint64_t wordE = 0x0000000000000000ULL;

        int iBit = 0;
        boost::uint64_t iDecision = 0ULL;

        for (std::vector<bool>::const_iterator itBit = m_gtDecisionWordExtended.begin();
                itBit != m_gtDecisionWordExtended.end(); ++itBit) {

            iDecision = static_cast<boost::uint64_t> (*itBit);//(*itBit ? 1 : 0);
            wordE = wordE | (iDecision << iBit);

            iBit++;
            if (iBit >= word64Size) {
                break;
            }

        }

        word64 = word64 | (wordE << GtDecisionWordExtendedShift);

    }

}



// set the NoAlgo value from a 64-bits word,
// having the index iWord in the GTFE raw record
void L1GtFdlWord::setNoAlgo(const boost::uint64_t& word64, int iWord)
{
    if (iWord == NoAlgoWord) {
        m_noAlgo = (word64 & NoAlgoMask) >> NoAlgoShift;
    }

}

// set the NoAlgo value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setNoAlgoWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == NoAlgoWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_noAlgo)
                           << NoAlgoShift);
    }

}




// set the FinalOR value from a 64-bits word,
// having the index iWord in the GTFE raw record
void L1GtFdlWord::setFinalOR(const boost::uint64_t& word64, int iWord)
{
    if (iWord == FinalORWord) {
        m_finalOR = (word64 & FinalORMask) >> FinalORShift;
    }

}

// set the FinalOR value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setFinalORWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == FinalORWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_finalOR)
                           << FinalORShift);
    }

}


// set the LocalBxNr value from a 64-bits word,
// having the index iWord in the GTFE raw record
void L1GtFdlWord::setLocalBxNr(const boost::uint64_t& word64, int iWord)
{
    if (iWord == LocalBxNrWord) {
        m_localBxNr = (word64 & LocalBxNrMask) >> LocalBxNrShift;
    }

}

// set the LocalBxNr value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtFdlWord::setLocalBxNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == LocalBxNrWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_localBxNr)
                           << LocalBxNrShift);
    }

}



// reset the content of a L1GtFdlWord
void L1GtFdlWord::reset()
{

    m_boardId = 0;
    m_bxInEvent = 0;
    m_bxNr = 0;
    m_eventNr = 0;

    // technical triggers std::vector<bool>
    m_gtTechnicalTriggerWord.assign(
        L1GlobalTriggerReadoutSetup::NumberTechnicalTriggers, false);

    // decision word  std::vector<bool>
    m_gtDecisionWord.assign(
        L1GlobalTriggerReadoutSetup::NumberPhysTriggers, false);

    // extended decision word  std::vector<bool>
    m_gtDecisionWordExtended.assign(
        L1GlobalTriggerReadoutSetup::NumberPhysTriggersExtended, false);

    m_noAlgo = 0;
    m_finalOR = 0;

    m_localBxNr = 0;

}

// pretty print the content of a L1GtFdlWord
void L1GtFdlWord::print(std::ostream& myCout) const
{

    myCout << "\n L1GtFdlWord::print \n" << std::endl;

    myCout << "  Board Id:         "
    << std::hex << " hex: " << "    " << std::setw(4) << std::setfill('0') << m_boardId
    << std::setfill(' ')
    << std::dec << " dec: " << m_boardId
    << std::endl;
    //

    int baseValue = 16; // using hexadecimal values;
    int hexBxInEvent = (m_bxInEvent + baseValue)%baseValue;

    myCout << "  BxInEvent:        "
    << std::hex << " hex: " << "       " << std::setw(1) << hexBxInEvent
    << std::dec << " dec: " << m_bxInEvent
    << std::endl;

    myCout << "  BxNr:             "
    << std::hex << " hex: "  << "     " << std::setw(3) << std::setfill('0') << m_bxNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_bxNr
    << std::endl;


    myCout << "  EventNr:          "
    << std::hex << " hex: " << "  " << std::setw(6) << std::setfill('0') << m_eventNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_eventNr
    << std::endl;

    myCout << "  TechnicalTrigger: " << std::endl;
    printGtTechnicalTriggerWord(myCout);

    // decision word (in two 64bits words)
    myCout << "  DecisionWord:     " << std::endl;

    int sizeW64 = 64;
    int iBit = 0;
    for (std::vector<bool>::const_reverse_iterator ritBit = m_gtDecisionWord.rbegin();
            ritBit != m_gtDecisionWord.rend(); ++ritBit) {

        myCout << (*ritBit ? '1' : '0');

        if (iBit == (sizeW64 - 1)) {
            myCout << std::endl;
        }
        
        iBit++;
    }

    // decision word extended (64 bits)
    myCout << "  DecisionWordExt:  " << std::endl;
    for (std::vector<bool>::const_reverse_iterator ritBit = m_gtDecisionWordExtended.rbegin();
            ritBit != m_gtDecisionWordExtended.rend(); ++ritBit) {

        myCout << (*ritBit ? '1' : '0');

    }

    myCout << "  NoAlgo:           "
    << std::hex << " hex: "  << "       " << std::setw(1) << std::setfill('0') << m_noAlgo
    << std::setfill(' ')
    << std::dec << " dec: " << m_noAlgo
    << std::endl;

    myCout << "  FinalOR:          "
    << std::hex << " hex: "  << "      " << std::setw(2) << std::setfill('0') << m_finalOR
    << std::setfill(' ')
    << std::dec << " dec: " << m_finalOR
    << std::endl;

    myCout << "  LocalBxNr:        "
    << std::hex << " hex: "  << "     " << std::setw(3) << std::setfill('0') << m_localBxNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_localBxNr
    << std::endl;

}



// static class members


