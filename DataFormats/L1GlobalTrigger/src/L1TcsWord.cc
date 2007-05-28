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
#include <iostream>
#include <iomanip>
#include <boost/cstdint.hpp>

// user include files
//   base class

// forward declarations

// constructors

// empty constructor, all members set to zero;
L1TcsWord::L1TcsWord()
{

    m_daqNr = 0;
    m_triggerType = 0;
    m_status = 0;
    m_bxNr = 0;
    m_partTrigNr = 0;
    m_eventNr = 0;
    m_assignedPartitions = 0;
    m_partRunNr = 0;
    m_orbitNr = 0;

}

// constructor from unpacked values;
L1TcsWord::L1TcsWord(
    boost::uint16_t daqNrValue,
    boost::uint16_t triggerTypeValue,
    boost::uint16_t statusValue,
    boost::uint16_t bxNrValue,
    boost::uint32_t partTrigNrValue,
    boost::uint32_t eventNrValue,
    boost::uint32_t assignedPartitionsValue,
    boost::uint32_t partRunNrValue,
    boost::uint32_t orbitNrValue )
{

    m_daqNr = daqNrValue;
    m_triggerType = triggerTypeValue;
    m_status = statusValue;
    m_bxNr = bxNrValue;
    m_partTrigNr = partTrigNrValue;
    m_eventNr = eventNrValue;
    m_assignedPartitions = assignedPartitionsValue;
    m_partRunNr = partRunNrValue;
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

    if (m_daqNr != result.m_daqNr) {
        return false;
    }

    if (m_triggerType != result.m_triggerType) {
        return false;
    }

    if (m_status != result.m_status) {
        return false;
    }

    if (m_bxNr != result.m_bxNr) {
        return false;
    }

    if (m_partTrigNr != result.m_partTrigNr) {
        return false;
    }

    if (m_eventNr != result.m_eventNr) {
        return false;
    }

    if (m_assignedPartitions != result.m_assignedPartitions) {
        return false;
    }

    if (m_partRunNr != result.m_partRunNr) {
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


// set the DaqNr value from a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setDaqNr(const boost::uint64_t& word64, int iWord)
{

    if (iWord == DaqNrWord) {
        m_daqNr = (word64 & DaqNrMask) >> DaqNrShift;
    }

}

// set the DaqNr value in a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setDaqNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == DaqNrWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_daqNr) << DaqNrShift);
    }

}


// set the TriggerType value from a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setTriggerType(const boost::uint64_t& word64, int iWord)
{

    if (iWord == TriggerTypeWord) {
        m_triggerType = (word64 & TriggerTypeMask) >> TriggerTypeShift;
    }

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


// set the Status value from a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setStatus(const boost::uint64_t& word64, int iWord)
{

    if (iWord == StatusWord) {
        m_status = (word64 & StatusMask) >> StatusShift;
    }

}

// set the Status value in a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setStatusWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == StatusWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_status) << StatusShift);
    }

}


// set the BxNr value from a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setBxNr(const boost::uint64_t& word64, int iWord)
{

    if (iWord == BxNrWord) {
        m_bxNr = (word64 & BxNrMask) >> BxNrShift;
    }

}

// set the BxNr value in a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setBxNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == BxNrWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_bxNr) << BxNrShift);
    }

}


// set the PartTrigNr value from a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setPartTrigNr(const boost::uint64_t& word64, int iWord)
{

    if (iWord == PartTrigNrWord) {
        m_partTrigNr = (word64 & PartTrigNrMask) >> PartTrigNrShift;
    }

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


// set the EventNr value from a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setEventNr(const boost::uint64_t& word64, int iWord)
{

    if (iWord == EventNrWord) {
        m_eventNr = (word64 & EventNrMask) >> EventNrShift;
    }

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


// set the AssignedPartitions value from a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setAssignedPartitions(const boost::uint64_t& word64, int iWord)
{

    if (iWord == AssignedPartitionsWord) {
        m_assignedPartitions = (word64 & AssignedPartitionsMask)
                               >> AssignedPartitionsShift;
    }

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


// set the PartRunNr value from a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setPartRunNr(const boost::uint64_t& word64, int iWord)
{

    if (iWord == PartRunNrWord) {
        m_partRunNr = (word64 & PartRunNrMask) >> PartRunNrShift;
    }

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


// set the OrbitNr value from a 64-bits word, having the index iWord
// in the GTFE EVM raw record
void L1TcsWord::setOrbitNr(const boost::uint64_t& word64, int iWord)
{

    if (iWord == OrbitNrWord) {
        m_orbitNr = (word64 & OrbitNrMask) >> OrbitNrShift;
    }

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

    m_daqNr = 0;
    m_triggerType = 0;
    m_status = 0;
    m_bxNr = 0;
    m_partTrigNr = 0;
    m_eventNr = 0;
    m_assignedPartitions = 0;
    m_partRunNr = 0;
    m_orbitNr = 0;

}


// pretty print
void L1TcsWord::print(std::ostream& myCout) const
{

    myCout << "\n L1TcsWord::print \n" << std::endl;

//    myCout << "  Board Id:  "
//    << std::hex << " hex:     " << std::setw(4) << std::setfill('0') << m_boardId
//    << std::setfill(' ')
//    << std::dec << " dec: " << m_boardId
//    << std::endl;


    myCout << "  DaqNr:              "
    << std::hex << " hex: " << "       " << std::setw(1) << m_daqNr
    << std::dec << " dec: " << m_daqNr
    << std::endl;

    myCout << "  TriggerType:        "
    << std::hex << " hex: " << "       " << std::setw(1) << m_triggerType
    << std::dec << " dec: " << m_triggerType
    << std::endl;

    myCout << "  Status:             "
    << std::hex << " hex: " << "       " << std::setw(1) << m_status
    << std::dec << " dec: " << m_status
    << std::endl;

    myCout << "  BxNr:               "
    << std::hex << " hex: "  << "     " << std::setw(3) << std::setfill('0') << m_bxNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_bxNr
    << std::endl;


    myCout << "  PartTrigNr:         "
    << std::hex << " hex: " << std::setw(8) << std::setfill('0') << m_partTrigNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_partTrigNr
    << std::endl;

    myCout << std::endl;
    
    myCout << "  EventNr:            "
    << std::hex << " hex: " << "  " << std::setw(6) << std::setfill('0') << m_eventNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_eventNr
    << std::endl;

    myCout << "  AssignedPartitions: "
    << std::hex << " hex: " << std::setw(8) << std::setfill('0') << m_assignedPartitions
    << std::setfill(' ')
    << std::dec << " dec: " << m_assignedPartitions
    << std::endl;

    myCout << std::endl;

    myCout << "  PartRunNr:          "
    << std::hex << " hex: " << std::setw(8) << std::setfill('0') << m_partTrigNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_partTrigNr
    << std::endl;
    
    myCout << "  OrbitNr:            "
    << std::hex << " hex: " << std::setw(8) << std::setfill('0') << m_orbitNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_orbitNr
    << std::endl;
    

}
