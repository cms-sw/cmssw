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
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"

// system include files
#include <iostream>
#include <iomanip>
#include <boost/cstdint.hpp>

// user include files
#include "FWCore/Utilities/interface/EDMException.h"


// constructors

// empty constructor, all members set to zero;
L1GtfeWord::L1GtfeWord()
{

    m_boardId = 0;
    m_recordLength = 0;
    m_bxNr = 0;
    m_setupVersion = 0;
    m_activeBoards = 0;
    m_totalTriggerNr = 0;

}

// constructor from unpacked values;
L1GtfeWord::L1GtfeWord(
    boost::uint16_t boardIdValue,
    boost::uint16_t recordLengthValue,
    boost::uint16_t bxNrValue,
    boost::uint32_t setupVersionValue,
    boost::uint16_t activeBoardsValue,
    boost::uint32_t totalTriggerNrValue)
{

    m_boardId = boardIdValue;
    m_recordLength = recordLengthValue;
    m_bxNr = bxNrValue;
    m_setupVersion = setupVersionValue;
    m_activeBoards = activeBoardsValue;
    m_totalTriggerNr = totalTriggerNrValue;

}

// destructor
L1GtfeWord::~L1GtfeWord()
{

    // empty now

}

// equal operator
bool L1GtfeWord::operator==(const L1GtfeWord& result) const
{

    if(m_boardId != result.m_boardId) {
        return false;
    }

    if(m_recordLength != result.m_recordLength) {
        return false;
    }

    if(m_bxNr != result.m_bxNr) {
        return false;
    }

    if(m_setupVersion != result.m_setupVersion) {
        return false;
    }

    if(m_activeBoards != result.m_activeBoards) {
        return false;
    }

    if(m_totalTriggerNr != result.m_totalTriggerNr) {
        return false;
    }

    // all members identical
    return true;

}

// unequal operator
bool L1GtfeWord::operator!=(const L1GtfeWord& result) const
{

    return !( result == *this);

}

// methods

// set the BoardId value from a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setBoardId(const boost::uint64_t& word64, int iWord)
{
    if (iWord == BoardIdWord) {
        m_boardId = (word64 & BoardIdMask) >> BoardIdShift;
    }

}

// set the BoardId value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setBoardIdWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == BoardIdWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_boardId) << BoardIdShift);
    }

}


// set the RecordLength value from a 64-bits word, having the index iWord in the GTFE raw record
void L1GtfeWord::setRecordLength(const boost::uint64_t& word64, int iWord)
{
    if (iWord == RecordLengthWord) {
        m_recordLength = (word64 & RecordLengthMask) >> RecordLengthShift;
    }

}

// set the RecordLength value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setRecordLengthWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == RecordLengthWord) {
        word64 = word64 |
                 (static_cast<boost::uint64_t> (m_recordLength) << RecordLengthShift);
    }

}

// set the BxNr value from a 64-bits word, having the index iWord in the GTFE raw record
void L1GtfeWord::setBxNr(const boost::uint64_t& word64, int iWord)
{
    if (iWord == BxNrWord) {
        m_bxNr = (word64 & BxNrMask) >> BxNrShift;
    }

}

// set the BxNr value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setBxNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == BxNrWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_bxNr) << BxNrShift);
    }

}


// set the SetupVersion value from a 64-bits word,
// having the index iWord in the GTFE raw record
void L1GtfeWord::setSetupVersion(const boost::uint64_t& word64, int iWord)
{
    if (iWord == SetupVersionWord) {
        m_setupVersion = (word64 & SetupVersionMask) >> SetupVersionShift;
    }

}

// set the SetupVersion value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setSetupVersionWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == SetupVersionWord) {
        word64 = word64 |
                 (static_cast<boost::uint64_t> (m_setupVersion) << SetupVersionShift);
    }

}

// set the ActiveBoards value from a 64-bits word,
// having the index iWord in the GTFE raw record
void L1GtfeWord::setActiveBoards(const boost::uint64_t& word64, int iWord)
{
    if (iWord == ActiveBoardsWord) {
        m_activeBoards = (word64 & ActiveBoardsMask) >> ActiveBoardsShift;
    }

}

// set the ActiveBoards value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setActiveBoardsWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == ActiveBoardsWord) {
        word64 = word64 |
                 (static_cast<boost::uint64_t> (m_activeBoards) << ActiveBoardsShift);
    }

}

// set the ActiveBoards value in a 64-bits word, having the index iWord
// in the GTFE raw record from the value activeBoardsValue
void L1GtfeWord::setActiveBoardsWord64(boost::uint64_t& word64, int iWord,
                                       boost::int16_t activeBoardsValue)
{
    if (iWord == ActiveBoardsWord) {
        word64 = word64 |
                 (static_cast<boost::uint64_t> (activeBoardsValue) << ActiveBoardsShift);
    }

}

// set the TotalTriggerNr value from a 64-bits word, having the index iWord in the GTFE raw record
void L1GtfeWord::setTotalTriggerNr(const boost::uint64_t& word64, int iWord)
{
    if (iWord == TotalTriggerNrWord) {
        m_totalTriggerNr = (word64 & TotalTriggerNrMask) >> TotalTriggerNrShift;
    }

}

// set the TotalTriggerNr value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeWord::setTotalTriggerNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == TotalTriggerNrWord) {
        word64 = word64 |
                 (static_cast<boost::uint64_t> (m_totalTriggerNr) << TotalTriggerNrShift);
    }

}



// reset the content of a L1GtfeWord
void L1GtfeWord::reset()
{

    m_boardId = 0;
    m_recordLength = 0;
    m_bxNr = 0;
    m_setupVersion = 0;
    //
    m_activeBoards = 0;
    m_totalTriggerNr = 0;
}

// pretty print the content of a L1GtfeWord
void L1GtfeWord::print(std::ostream& myCout) const
{

    myCout << "\n L1GtfeWord::print \n" << std::endl;

    myCout << "  BoardId:        "
    << std::hex << " hex: " << "      " << std::setw(2) << std::setfill('0') << m_boardId
    << std::setfill(' ')
    << std::dec << " dec: " << m_boardId
    << std::endl;

    myCout << "  RecordLength:   "
    << std::hex << " hex: " << "       " << std::setw(1) <<  m_recordLength
    << std::dec << " dec: " <<  m_recordLength
    << std::endl;

    myCout << "  BxNr:           "
    << std::hex << " hex: "  << "     " << std::setw(3) << std::setfill('0') << m_bxNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_bxNr
    << std::endl;

    myCout << "  SetupVersion:   "
    << std::hex << " hex: " << std::setw(8) << std::setfill('0') << m_setupVersion
    << std::setfill(' ')
    << std::dec << " dec: " << m_setupVersion
    << std::endl;

    //

    myCout << "  ActiveBoards:   "
    << std::hex << " hex: " << "    " << std::setw(4) << std::setfill('0') << m_activeBoards
    << std::setfill(' ')
    << std::dec << " dec: " << m_activeBoards
    << std::endl;

    myCout << "  TotalTriggerNr: "
    << std::hex << " hex: " << std::setw(8) << std::setfill('0') << m_totalTriggerNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_totalTriggerNr
    << std::endl;


}

// static class members
const int L1GtfeWord::BlockSize;        // 2 x 64bits

const int L1GtfeWord::BoardIdWord;
const int L1GtfeWord::RecordLengthWord;
const int L1GtfeWord::BxNrWord;
const int L1GtfeWord::SetupVersionWord;

const boost::uint64_t L1GtfeWord::BoardIdMask;
const boost::uint64_t L1GtfeWord::RecordLengthMask;
const boost::uint64_t L1GtfeWord::BxNrMask;
const boost::uint64_t L1GtfeWord::SetupVersionMask;

const int L1GtfeWord::BoardIdShift;
const int L1GtfeWord::RecordLengthShift;
const int L1GtfeWord::BxNrShift;
const int L1GtfeWord::SetupVersionShift;

//
const int L1GtfeWord::ActiveBoardsWord;
const int L1GtfeWord::TotalTriggerNrWord;

const boost::uint64_t L1GtfeWord::ActiveBoardsMask;
const boost::uint64_t L1GtfeWord::TotalTriggerNrMask;

const int L1GtfeWord::ActiveBoardsShift;
const int L1GtfeWord::TotalTriggerNrShift;
