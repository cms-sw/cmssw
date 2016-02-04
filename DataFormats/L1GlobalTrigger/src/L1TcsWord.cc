/**
 * \class L1TcsWord
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
#include "DataFormats/L1GlobalTrigger/interface/L1TcsWord.h"

// system include files
#include <iomanip>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// forward declarations

// constructors

// empty constructor, all members set to zero;
L1TcsWord::L1TcsWord()
{

    m_boardId = 0;
    m_bxNr = 0;
    m_daqNr = 0;
    m_triggerType = 0;
    m_status = 0;
    m_luminositySegmentNr = 0;
    m_partRunNr = 0;
    m_assignedPartitions = 0;
    m_partTrigNr = 0;
    m_eventNr = 0;
    m_orbitNr = 0;

}

// constructor from unpacked values;
L1TcsWord::L1TcsWord(
    boost::uint16_t boardIdValue,
    boost::uint16_t bxNrValue,
    boost::uint16_t daqNrValue,
    boost::uint16_t triggerTypeValue,
    boost::uint16_t statusValue,
    boost::uint16_t luminositySegmentNrValue,
    boost::uint32_t partRunNrValue,
    boost::uint32_t assignedPartitionsValue,
    boost::uint32_t partTrigNrValue,
    boost::uint32_t eventNrValue,
    boost::uint64_t orbitNrValue )
{

    m_boardId = boardIdValue;
    m_bxNr = bxNrValue;
    m_daqNr = daqNrValue;
    m_triggerType = triggerTypeValue;
    m_status = statusValue;
    m_luminositySegmentNr = luminositySegmentNrValue;
    m_partRunNr = partRunNrValue;
    m_assignedPartitions = assignedPartitionsValue;
    m_partTrigNr = partTrigNrValue;
    m_eventNr = eventNrValue;
    m_orbitNr = orbitNrValue;

}

// destructor
L1TcsWord::~L1TcsWord()
{

    // empty

}

// equal operator
bool L1TcsWord::operator==(const L1TcsWord& result) const
{

    if (m_boardId != result.m_boardId) {
        return false;
    }

    if (m_bxNr != result.m_bxNr) {
        return false;
    }

    if (m_daqNr != result.m_daqNr) {
        return false;
    }

    if (m_triggerType != result.m_triggerType) {
        return false;
    }

    if (m_status != result.m_status) {
        return false;
    }

    if (m_luminositySegmentNr != result.m_luminositySegmentNr) {
        return false;
    }

    if (m_partRunNr != result.m_partRunNr) {
        return false;
    }

    if (m_assignedPartitions != result.m_assignedPartitions) {
        return false;
    }

    if (m_partTrigNr != result.m_partTrigNr) {
        return false;
    }

    if (m_eventNr != result.m_eventNr) {
        return false;
    }

    if (m_orbitNr != result.m_orbitNr) {
        return false;
    }

    // all members identical
    return true;

}

// unequal operator
bool L1TcsWord::operator!=(const L1TcsWord& result) const
{

    return !( result == *this);

}

// methods

// set the BoardId value from a 64-bits word
void L1TcsWord::setBoardId(const boost::uint64_t& word64)
{
    m_boardId = (word64 & BoardIdMask) >> BoardIdShift;

}

// set the BoardId value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1TcsWord::setBoardIdWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == BoardIdWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_boardId) << BoardIdShift);
    }

}


// set the BxNr value from a 64-bits word
void L1TcsWord::setBxNr(const boost::uint64_t& word64)
{

    m_bxNr = (word64 & BxNrMask) >> BxNrShift;

}

// set the BxNr value in a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setBxNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == BxNrWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_bxNr) << BxNrShift);
    }

}



// set the DaqNr value from a 64-bits word
void L1TcsWord::setDaqNr(const boost::uint64_t& word64)
{

    m_daqNr = (word64 & DaqNrMask) >> DaqNrShift;

}

// set the DaqNr value in a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setDaqNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == DaqNrWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_daqNr) << DaqNrShift);
    }

}


// set the TriggerType value from a 64-bits word
void L1TcsWord::setTriggerType(const boost::uint64_t& word64)
{

    m_triggerType = (word64 & TriggerTypeMask) >> TriggerTypeShift;

}

// set the TriggerType value in a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setTriggerTypeWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == TriggerTypeWord) {
        word64 = word64 |
                 (static_cast<boost::uint64_t> (m_triggerType) << TriggerTypeShift);
    }

}


// set the Status value from a 64-bits word
void L1TcsWord::setStatus(const boost::uint64_t& word64)
{

    m_status = (word64 & StatusMask) >> StatusShift;

}

// set the Status value in a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setStatusWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == StatusWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_status) << StatusShift);
    }

}



// set the luminosity segment number value from a 64-bits word
void L1TcsWord::setLuminositySegmentNr(const boost::uint64_t& word64)
{

    m_luminositySegmentNr =
        (word64 & LuminositySegmentNrMask) >> LuminositySegmentNrShift;

}

// set the luminosity segment number value in a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setLuminositySegmentNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == LuminositySegmentNrWord) {
        word64 = word64 |
                 (static_cast<boost::uint64_t> (m_luminositySegmentNr) << LuminositySegmentNrShift);
    }

}



// set the PartRunNr value from a 64-bits word
void L1TcsWord::setPartRunNr(const boost::uint64_t& word64)
{

    m_partRunNr = (word64 & PartRunNrMask) >> PartRunNrShift;

}

// set the PartRunNr value in a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setPartRunNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == PartRunNrWord) {
        word64 = word64 |
                 (static_cast<boost::uint64_t> (m_partRunNr) << PartRunNrShift);
    }

}



// set the AssignedPartitions value from a 64-bits word
void L1TcsWord::setAssignedPartitions(const boost::uint64_t& word64)
{

    m_assignedPartitions = (word64 & AssignedPartitionsMask)
                           >> AssignedPartitionsShift;

}

// set the AssignedPartitions value in a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setAssignedPartitionsWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == AssignedPartitionsWord) {
        word64 = word64 |
                 (static_cast<boost::uint64_t> (m_assignedPartitions)
                  << AssignedPartitionsShift);
    }

}



// set the PartTrigNr value from a 64-bits word
void L1TcsWord::setPartTrigNr(const boost::uint64_t& word64)
{

    m_partTrigNr = (word64 & PartTrigNrMask) >> PartTrigNrShift;

}

// set the PartTrigNr value in a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setPartTrigNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == PartTrigNrWord) {
        word64 = word64 |
                 (static_cast<boost::uint64_t> (m_partTrigNr) << PartTrigNrShift);
    }

}



// set the EventNr value from a 64-bits word
void L1TcsWord::setEventNr(const boost::uint64_t& word64)
{

    m_eventNr = (word64 & EventNrMask) >> EventNrShift;

}

// set the EventNr value in a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setEventNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == EventNrWord) {
        word64 = word64 |
                 (static_cast<boost::uint64_t> (m_eventNr) << EventNrShift);
    }

}



// set the OrbitNr value from a 64-bits word
void L1TcsWord::setOrbitNrFrom(const boost::uint64_t& word64)
{

    m_orbitNr = (word64 & OrbitNrMask) >> OrbitNrShift;

}

// set the OrbitNr value in a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setOrbitNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == OrbitNrWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_orbitNr) << OrbitNrShift);
    }

}


// reset the content of a L1TcsWord
void L1TcsWord::reset()
{

    m_boardId = 0;
    m_bxNr = 0;
    m_daqNr = 0;
    m_triggerType = 0;
    m_status = 0;
    m_luminositySegmentNr = 0;
    m_partRunNr = 0;
    m_assignedPartitions = 0;
    m_partTrigNr = 0;
    m_eventNr = 0;
    m_orbitNr = 0;


}


// pretty print
void L1TcsWord::print(std::ostream& myCout) const
{

    myCout << "\n L1TcsWord::print \n" << std::endl;

    int iWord = 0;

    myCout << "\n Word " << iWord << std::endl;

    myCout << "  Board Id:            "
    << std::hex << " hex: " << "        "<< std::setw(4) << std::setfill('0') << m_boardId
    << std::setfill(' ')
    << std::dec << " dec: " << m_boardId
    << std::endl;


    myCout << "  BxNr:                "
    << std::hex << " hex: "  << "         " << std::setw(3) << std::setfill('0') << m_bxNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_bxNr
    << std::endl;


    myCout << "  DaqNr:               "
    << std::hex << " hex: " << "           " << std::setw(1) << m_daqNr
    << std::dec << " dec: " << m_daqNr
    << std::endl;

    myCout << "  TriggerType:         "
    << std::hex << " hex: " << "           " << std::setw(1) << m_triggerType
    << std::dec << " dec: " << m_triggerType
    << std::endl;

    myCout << "  Status:              "
    << std::hex << " hex: " << "           " << std::setw(1) << m_status
    << std::dec << " dec: " << m_status
    << std::endl;

    myCout << "  LuminositySegmentNr: "
    << std::hex << " hex: " << "        " << std::setw(4) << m_luminositySegmentNr
    << std::dec << " dec: " << m_luminositySegmentNr
    << std::endl;


    iWord++;
    myCout << "\n Word " << iWord << std::endl;

    myCout << "  PartRunNr:           "
    << std::hex << " hex: " << "    " << std::setw(8) << std::setfill('0') << m_partRunNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_partRunNr
    << std::endl;

    myCout << "  AssignedPartitions:  "
    << std::hex << " hex: " << "    " << std::setw(8) << std::setfill('0')
    << m_assignedPartitions
    << std::setfill(' ')
    << std::dec << " dec: " << m_assignedPartitions
    << std::endl;


    iWord++;
    myCout << "\n Word " << iWord << std::endl;

    myCout << "  PartTrigNr:          "
    << std::hex << " hex: " << "    " << std::setw(8) << std::setfill('0') << m_partTrigNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_partTrigNr
    << std::endl;


    myCout << "  EventNr:             "
    << std::hex << " hex: " << "    " << std::setw(8) << std::setfill('0') << m_eventNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_eventNr
    << std::endl;

    iWord++;
    myCout << "\n Word " << iWord << std::endl;

    myCout << "  OrbitNr:             "
    << std::hex << " hex: " << "" << std::setw(12) << std::setfill('0') << m_orbitNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_orbitNr
    << std::endl;

    iWord++;
    myCout << "\n Word " << iWord << std::endl;

    myCout << "  Empty word          "
    << std::endl;

}


// unpack TCS
// tcsPtr pointer to the beginning of the TCS block in the raw data
void L1TcsWord::unpack(const unsigned char* tcsPtr)
{
    LogDebug("L1GtTcsWord")
    << "\nUnpacking TCS block.\n"
    << std::endl;

    const boost::uint64_t* payload =
        reinterpret_cast<boost::uint64_t*>(const_cast<unsigned char*>(tcsPtr));

    setBoardId(payload[BoardIdWord]);
    setBxNr(payload[BxNrWord]);
    setDaqNr(payload[DaqNrWord]);
    setTriggerType(payload[TriggerTypeWord]);
    setStatus(payload[StatusWord]);
    setLuminositySegmentNr(payload[LuminositySegmentNrWord]);

    setPartRunNr(payload[PartRunNrWord]);
    setAssignedPartitions(payload[AssignedPartitionsWord]);

    setPartTrigNr(payload[PartTrigNrWord]);
    setEventNr(payload[EventNrWord]);

    setOrbitNrFrom(payload[OrbitNrWord]);

    if ( edm::isDebugEnabled() ) {

        for (int iWord = 0; iWord < BlockSize; ++iWord) {
            LogTrace("L1GtTcsWord")
            << std::setw(4) << iWord << "  "
            << std::hex << std::setfill('0')
            << std::setw(16) << payload[iWord]
            << std::dec << std::setfill(' ')
            << std::endl;
        }

    }

}

// static class members

// block description in the raw GT record

// block size in 64bits words (BlockSize * 64 bits)
const int L1TcsWord::BlockSize = 5;

// word 0

// index of the word in the TCS block containig the variable
const int L1TcsWord::BoardIdWord = 0;
const int L1TcsWord::BxNrWord = 0;
const int L1TcsWord::DaqNrWord = 0;
const int L1TcsWord::TriggerTypeWord = 0;
const int L1TcsWord::StatusWord = 0;
const int L1TcsWord::LuminositySegmentNrWord = 0;

// mask to get the 64-bit-value from the corresponding word in the TCS block
const boost::uint64_t L1TcsWord::BoardIdMask =               0xFFFF000000000000ULL;
const boost::uint64_t L1TcsWord::BxNrMask =                  0x00000FFF00000000ULL;
const boost::uint64_t L1TcsWord::DaqNrMask =                 0x000000000F000000ULL;
const boost::uint64_t L1TcsWord::TriggerTypeMask =           0x0000000000F00000ULL;
const boost::uint64_t L1TcsWord::StatusMask =                0x00000000000F0000ULL;
const boost::uint64_t L1TcsWord::LuminositySegmentNrMask =   0x000000000000FFFFULL;

// shift to the right to get the value from the "64-bit-value"
const int L1TcsWord::BoardIdShift = 48;
const int L1TcsWord::BxNrShift = 32;
const int L1TcsWord::DaqNrShift = 24;
const int L1TcsWord::TriggerTypeShift = 20;
const int L1TcsWord::StatusShift = 16;
const int L1TcsWord::LuminositySegmentNrShift = 0;

// word 1

const int L1TcsWord::PartRunNrWord = 1;
const int L1TcsWord::AssignedPartitionsWord = 1;

const boost::uint64_t L1TcsWord::PartRunNrMask          =   0xFFFFFFFF00000000ULL;
const boost::uint64_t L1TcsWord::AssignedPartitionsMask =   0x00000000FFFFFFFFULL;

const int L1TcsWord::PartRunNrShift = 32;
const int L1TcsWord::AssignedPartitionsShift = 0;

// word 2

const int L1TcsWord::PartTrigNrWord = 2;
const int L1TcsWord::EventNrWord = 2;

const boost::uint64_t L1TcsWord::PartTrigNrMask =   0xFFFFFFFF00000000ULL;
const boost::uint64_t L1TcsWord::EventNrMask    =   0x00000000FFFFFFFFULL;

const int L1TcsWord::PartTrigNrShift = 32;
const int L1TcsWord::EventNrShift = 0;

// word 3

const int L1TcsWord::OrbitNrWord = 3;

const boost::uint64_t L1TcsWord::OrbitNrMask =   0x0000FFFFFFFFFFFFULL;

const int L1TcsWord::OrbitNrShift = 0;


// word 4

// empty



