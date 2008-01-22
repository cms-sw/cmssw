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
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"

// system include files
#include <iomanip>

// user include files
#include "FWCore/Utilities/interface/EDMException.h"


// constructors

// empty constructor, all members set to zero;
L1GtPsbWord::L1GtPsbWord()
{

    m_boardId = 0;
    m_bxInEvent = 0;
    m_bxNr = 0;
    m_eventNr = 0;

    for (int iA = 0; iA < NumberAData; ++iA) {
        m_aData[iA] = 0;
    }

    for (int iB = 0; iB < NumberBData; ++iB) {
        m_bData[iB] = 0;
    }

    m_localBxNr = 0;

}

// constructor from unpacked values;
L1GtPsbWord::L1GtPsbWord(
    boost::uint16_t boardIdValue,
    int bxInEventValue,
    boost::uint16_t bxNrValue,
    boost::uint32_t eventNrValue,
    boost::uint16_t aDataValue[NumberAData],
    boost::uint16_t bDataValue[NumberBData],
    boost::uint16_t localBxNrValue
)
{

    m_boardId = boardIdValue;
    m_bxInEvent = bxInEventValue;
    m_bxNr = bxNrValue;
    m_eventNr = eventNrValue;

    for (int iA = 0; iA < NumberAData; ++iA) {
        m_aData[iA] = aDataValue[iA];
    }

    for (int iB = 0; iB < NumberBData; ++iB) {
        m_bData[iB] = bDataValue[iB];
    }

    m_localBxNr = localBxNrValue;

}

// destructor
L1GtPsbWord::~L1GtPsbWord()
{

    // empty now

}

// equal operator
bool L1GtPsbWord::operator==(const L1GtPsbWord& result) const
{

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

    for (int iA = 0; iA < NumberAData; ++iA) {        
        if (m_aData[iA] != result.m_aData[iA]) {
            return false;
        }    
    }

    for (int iB = 0; iB < NumberBData; ++iB) {
        if (m_bData[iB] != result.m_bData[iB]) {
            return false;
        }
    }
    
    if (m_localBxNr != result.m_localBxNr) {
        return false;
    }

    // all members identical
    return true;

}

// unequal operator
bool L1GtPsbWord::operator!=(const L1GtPsbWord& result) const
{

    return !( result == *this);

}


// methods

// set the BoardId value from a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtPsbWord::setBoardId(const boost::uint64_t& word64, int iWord)
{
    if (iWord == BoardIdWord) {
        m_boardId = (word64 & BoardIdMask) >> BoardIdShift;
    }

}

// set the BoardId value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtPsbWord::setBoardIdWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == BoardIdWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_boardId) << BoardIdShift);
    }

}


// set the BxInEvent value from a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtPsbWord::setBxInEvent(const boost::uint64_t& word64, int iWord)
{
    if (iWord == BxInEventWord) {
        int baseValue = 16; // using hexadecimal values;
        int hexBxInEvent = (word64 & BxInEventMask) >> BxInEventShift;
        m_bxInEvent = (hexBxInEvent + baseValue/2)%baseValue - baseValue/2;
    }

}

// set the BxInEvent value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtPsbWord::setBxInEventWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == BxInEventWord) {
        int baseValue = 16; // using hexadecimal values;
        int hexBxInEvent = (m_bxInEvent + baseValue)%baseValue;
        word64 = word64 | (static_cast<boost::uint64_t> (hexBxInEvent)
                           << BxInEventShift);
    }

}


// set the BxNr value from a 64-bits word, having the index iWord in the GTFE raw record
void L1GtPsbWord::setBxNr(const boost::uint64_t& word64, int iWord)
{

    if (iWord == BxNrWord) {
        m_bxNr = (word64 & BxNrMask) >> BxNrShift;
    }

}

// set the BxNr value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtPsbWord::setBxNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == BxNrWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_bxNr) << BxNrShift);
    }

}


// set the EventNr value from a 64-bits word, having the index iWord in the GTFE raw record
void L1GtPsbWord::setEventNr(const boost::uint64_t& word64, int iWord)
{
    if (iWord == EventNrWord) {
        m_eventNr = (word64 & EventNrMask) >> EventNrShift;
    }

}

// set the EventNr value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtPsbWord::setEventNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == EventNrWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_eventNr) << EventNrShift);
    }

}


// get/set A_DATA_CH_IA

const boost::uint16_t L1GtPsbWord::aData(int iA) const
{

    if (iA < 0 || iA > NumberAData) {
        throw cms::Exception("aDataIndexError")
        << "\nError: index for A_DATA array out of range. Allowed range: [0, "
        << NumberAData << ") " << std::endl;

    } else {
        return m_aData[iA];
    }

}

void L1GtPsbWord::setAData(boost::uint16_t aDataVal, int iA)
{

    if (iA < 0 || iA > NumberAData) {
        throw cms::Exception("aDataIndexError")
        << "\nError: index for A_DATA array out of range. Allowed range: [0, "
        << NumberAData << ") " << std::endl;

    } else {
        m_aData[iA] = aDataVal;
    }

}

// set the AData value from a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtPsbWord::setAData(const boost::uint64_t& word64, int iWord)
{

    int sizeW64 = sizeof(word64)*8;
    int nSubWords = sizeW64/DataCHSize;

    if (iWord == ADataCH0Word) {

        for (int i = 0; i < nSubWords; ++i) {
            int dataShift = i*DataCHSize;
            m_aData[i] = (word64 & (DataCHMask << dataShift)) >> dataShift;

            //            LogTrace("L1GtPsbWord")
            //            << "\n  A_Data_CH" << i << " = "
            //            << m_aData[i]
            //            << std::endl;

        }

    } else if (iWord == ADataCH4Word) {

        for (int i = 0; i < nSubWords; ++i) {
            int dataShift = i*DataCHSize;
            m_aData[i + nSubWords] = (word64 & (DataCHMask << dataShift)) >> dataShift;

            //            LogTrace("L1GtPsbWord")
            //            << "\n  A_Data_CH" << i + nSubWords << " = "
            //            << m_aData[i]
            //            << std::endl;
        }

    }

}

// set the AData value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtPsbWord::setADataWord64(boost::uint64_t& word64, int iWord)
{

    int sizeW64 = sizeof(word64)*8;
    int nSubWords = sizeW64/DataCHSize;

    if (iWord == ADataCH0Word) {

        for (int i = 0; i < nSubWords; ++i) {
            int dataShift = i*DataCHSize;
            word64 = word64 | (static_cast<boost::uint64_t> (m_aData[i]) << dataShift);
        }

    }
    else if (iWord == ADataCH4Word) {

        for (int i = 0; i < nSubWords; ++i) {
            int dataShift = i*DataCHSize;
            word64 = word64 |
                     (static_cast<boost::uint64_t> (m_aData[i + nSubWords]) << dataShift);
        }

    }

}

// get/set B_DATA_CH_IB

const boost::uint16_t L1GtPsbWord::bData(int iB) const
{

    if (iB < 0 || iB > NumberBData) {
        throw cms::Exception("bDataIndexError")
        << "\nError: index for B_DATA array out of range. Allowed range: [0, "
        << NumberBData << ") " << std::endl;

    } else {
        return m_bData[iB];
    }

}

void L1GtPsbWord::setBData(boost::uint16_t bDataVal, int iB)
{

    if (iB < 0 || iB > NumberBData) {
        throw cms::Exception("bDataIndexError")
        << "\nError: index for B_DATA array out of range. Allowed range: [0, "
        << NumberBData << ") " << std::endl;

    } else {
        m_bData[iB] = bDataVal;
    }

}

// set the BData value from a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtPsbWord::setBData(const boost::uint64_t& word64, int iWord)
{

    int sizeW64 = sizeof(word64)*8;
    int nSubWords = sizeW64/DataCHSize;

    if (iWord == BDataCH0Word) {

        for (int i = 0; i < nSubWords; ++i) {
            int dataShift = i*DataCHSize;
            m_bData[i] = (word64 & (DataCHMask << dataShift)) >> dataShift;
        }

    } else if (iWord == BDataCH4Word) {

        for (int i = 0; i < nSubWords; ++i) {
            int dataShift = i*DataCHSize;
            m_bData[i + nSubWords] = (word64 & (DataCHMask << dataShift)) >> dataShift;
        }

    }

}

// set the BData value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtPsbWord::setBDataWord64(boost::uint64_t& word64, int iWord)
{

    int sizeW64 = sizeof(word64)*8;
    int nSubWords = sizeW64/DataCHSize;

    if (iWord == BDataCH0Word) {

        for (int i = 0; i < nSubWords; ++i) {
            int dataShift = i*DataCHSize;
            word64 = word64 | (static_cast<boost::uint64_t> (m_bData[i]) << dataShift);
        }

    }
    else if (iWord == BDataCH4Word) {

        for (int i = 0; i < nSubWords; ++i) {
            int dataShift = i*DataCHSize;
            word64 = word64 |
                     (static_cast<boost::uint64_t> (m_bData[i + nSubWords]) << dataShift);
        }

    }

}


// set the LocalBxNr value from a 64-bits word,
// having the index iWord in the GTFE raw record
void L1GtPsbWord::setLocalBxNr(const boost::uint64_t& word64, int iWord)
{
    if (iWord == LocalBxNrWord) {
        m_localBxNr = (word64 & LocalBxNrMask) >> LocalBxNrShift;
    }

}

// set the LocalBxNr value in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtPsbWord::setLocalBxNrWord64(boost::uint64_t& word64, int iWord)
{

    if (iWord == LocalBxNrWord) {
        word64 = word64 | (static_cast<boost::uint64_t> (m_localBxNr)
                           << LocalBxNrShift);
    }

}




// reset the content of a L1GtPsbWord
void L1GtPsbWord::reset()
{

    m_boardId = 0;
    m_bxInEvent = 0;
    m_bxNr = 0;
    m_eventNr = 0;

    for (int iA = 0; iA < NumberAData; ++iA) {
        m_aData[iA] = 0;
    }

    for (int iB = 0; iB < NumberBData; ++iB) {
        m_bData[iB] = 0;
    }

    m_localBxNr = 0;

}

// pretty print
void L1GtPsbWord::print(std::ostream& myCout) const
{

    myCout << "\n L1GtPsbWord::print \n" << std::endl;

    myCout << "  Board Id:  "
    << std::hex << " hex:     " << std::setw(4) << std::setfill('0') << m_boardId
    << std::setfill(' ')
    << std::dec << " dec: " << m_boardId
    << std::endl;

    int baseValue = 16; // using hexadecimal values;
    int hexBxInEvent = (m_bxInEvent + baseValue)%baseValue;

    myCout << "  BxInEvent: "
    << std::hex << " hex:     " << "   " << std::setw(1) << hexBxInEvent
    << std::dec << " dec: " << m_bxInEvent
    << std::endl;

    myCout << "  BxNr:      "
    << std::hex << " hex:     "  << " " << std::setw(3) << std::setfill('0') << m_bxNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_bxNr
    << std::endl;


    myCout << "  EventNr:   "
    << std::hex << " hex: " << "  " << std::setw(6) << std::setfill('0') << m_eventNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_eventNr
    << std::endl;

    int sizeW64 = 64;
    int dataBlocksPerLine = sizeW64/DataCHSize; // 4x16 bits per line

    myCout << "\n        "
    << "A_Data_CH3 "
    << "A_Data_CH2 "
    << "A_Data_CH1 "
    << "A_Data_CH0 " << "\n"
    << std::hex << "  hex:  " << std::setfill('0');

    for (int i = 0; i < dataBlocksPerLine; ++i) {

        int iCh = dataBlocksPerLine - (i + 1); // reverse
        myCout << std::setw(4) <<  m_aData[iCh] << "       ";

    }

    myCout << "\n"
    << std::dec << "  dec:  ";

    for (int i = 0; i < dataBlocksPerLine; ++i) {

        int iCh = dataBlocksPerLine - (i + 1); // reverse
        myCout << std::setw(5) <<  m_aData[iCh] << "      ";

    }

    myCout << "\n\n        "
    << "A_Data_CH7 "
    << "A_Data_CH6 "
    << "A_Data_CH5 "
    << "A_Data_CH4 " << "\n"
    << std::hex << "  hex:  " << std::setfill('0');

    for (int i = 0; i < dataBlocksPerLine; ++i) {

        int iCh = dataBlocksPerLine - (i + 1); // reverse
        myCout << std::setw(4) <<  m_aData[iCh + dataBlocksPerLine] << "       ";

    }

    myCout << "\n"
    << std::dec << "  dec:  ";

    for (int i = 0; i < dataBlocksPerLine; ++i) {

        int iCh = dataBlocksPerLine - (i + 1); // reverse
        myCout << std::setw(5) <<  m_aData[iCh + dataBlocksPerLine] << "      ";

    }

    myCout << std::endl;

    myCout << "\n        "
    << "B_Data_CH3 "
    << "B_Data_CH2 "
    << "B_Data_CH1 "
    << "B_Data_CH0 " << "\n"
    << std::hex << "  hex:  " << std::setfill('0');

    for (int i = 0; i < dataBlocksPerLine; ++i) {

        int iCh = dataBlocksPerLine - (i + 1); // reverse
        myCout << std::setw(4) <<  m_bData[iCh] << "       ";

    }

    myCout << "\n"
    << std::dec << "  dec:  ";

    for (int i = 0; i < dataBlocksPerLine; ++i) {

        int iCh = dataBlocksPerLine - (i + 1); // reverse
        myCout << std::setw(5) <<  m_bData[iCh] << "      ";

    }

    myCout << "\n\n        "
    << "B_Data_CH7 "
    << "B_Data_CH6 "
    << "B_Data_CH5 "
    << "B_Data_CH4 " << "\n"
    << std::hex << "  hex:  " << std::setfill('0');

    for (int i = 0; i < dataBlocksPerLine; ++i) {

        int iCh = dataBlocksPerLine - (i + 1); // reverse
        myCout << std::setw(4) <<  m_bData[iCh + dataBlocksPerLine] << "       ";

    }

    myCout << "\n"
    << std::dec << "  dec:  ";

    for (int i = 0; i < dataBlocksPerLine; ++i) {

        int iCh = dataBlocksPerLine - (i + 1); // reverse
        myCout << std::setw(5) <<  m_bData[iCh + dataBlocksPerLine] << "      ";

    }

    myCout << "\n" << std::endl;

    myCout << "  LocalBxNr: "
    << std::hex << " hex:     "  << " " << std::setw(3) << std::setfill('0') << m_localBxNr
    << std::setfill(' ')
    << std::dec << " dec: " << m_localBxNr
    << std::endl;

}

// static class members
const int L1GtPsbWord::NumberAData;
const int L1GtPsbWord::NumberBData;

