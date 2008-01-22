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

// this class header
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"

// system include files
#include <iostream>
#include <iomanip>



// user include files
//   base class

// forward declarations

// constructors
L1GtBoard::L1GtBoard()
{

    // empty

}

L1GtBoard::L1GtBoard(const L1GtBoardType& gtBoardTypeValue)
{

    m_gtBoardType = gtBoardTypeValue;

    m_gtBoardIndex = -1;

    m_gtPositionDaqRecord = -1;
    m_gtPositionEvmRecord = -1;

    m_gtBitDaqActiveBoards = -1;
    m_gtBitEvmActiveBoards = -1;

    m_gtBoardSlot = -1;
    m_gtBoardHexName = 0;

    m_gtQuadInPsb.reserve(NumberCablesBoard);

}

L1GtBoard::L1GtBoard(const L1GtBoardType& gtBoardTypeValue, const int& gtBoardIndexValue)
{

    m_gtBoardType = gtBoardTypeValue;
    m_gtBoardIndex = gtBoardIndexValue;

    m_gtPositionDaqRecord = -1;
    m_gtPositionEvmRecord = -1;

    m_gtBitDaqActiveBoards = -1;
    m_gtBitEvmActiveBoards = -1;

    m_gtBoardSlot = -1;
    m_gtBoardHexName = 0;

    m_gtQuadInPsb.reserve(NumberCablesBoard);

}

// destructor
L1GtBoard::~L1GtBoard()
{
    // empty
}

// copy constructor
L1GtBoard::L1GtBoard(const L1GtBoard& gtb)
{

    m_gtBoardType = gtb.gtBoardType();
    m_gtBoardIndex = gtb.gtBoardIndex();

    m_gtPositionDaqRecord = gtb.gtPositionDaqRecord();
    m_gtPositionEvmRecord = gtb.gtPositionEvmRecord();

    m_gtBitDaqActiveBoards = gtb.gtBitDaqActiveBoards();
    m_gtBitEvmActiveBoards = gtb.gtBitEvmActiveBoards();

    m_gtBoardSlot = gtb.gtBoardSlot();
    m_gtBoardHexName = gtb.gtBoardHexName();

    m_gtQuadInPsb = gtb.gtQuadInPsb();

}

// assignment operator
L1GtBoard::L1GtBoard& L1GtBoard::operator=(const L1GtBoard& gtb)
{

    if ( this != &gtb ) {

        m_gtBoardType = gtb.gtBoardType();
        m_gtBoardIndex = gtb.gtBoardIndex();

        m_gtPositionDaqRecord = gtb.gtPositionDaqRecord();
        m_gtPositionEvmRecord = gtb.gtPositionEvmRecord();

        m_gtBitDaqActiveBoards = gtb.gtBitDaqActiveBoards();
        m_gtBitEvmActiveBoards = gtb.gtBitEvmActiveBoards();

        m_gtBoardSlot = gtb.gtBoardSlot();
        m_gtBoardHexName = gtb.gtBoardHexName();

        m_gtQuadInPsb = gtb.gtQuadInPsb();

    }

    return *this;

}

// equal operator
bool L1GtBoard::operator==(const L1GtBoard& gtb) const
{

    if (m_gtBoardType != gtb.gtBoardType()) {
        return false;
    }

    if (m_gtBoardIndex != gtb.gtBoardIndex()) {
        return false;
    }

    if (m_gtPositionDaqRecord != gtb.gtPositionDaqRecord()) {
        return false;
    }

    if (m_gtPositionEvmRecord != gtb.gtPositionEvmRecord()) {
        return false;
    }

    if (m_gtBitDaqActiveBoards != gtb.gtBitDaqActiveBoards()) {
        return false;
    }

    if (m_gtBitEvmActiveBoards != gtb.gtBitEvmActiveBoards()) {
        return false;
    }

    if (m_gtBoardSlot != gtb.gtBoardSlot()) {
        return false;
    }

    if (m_gtBoardHexName != gtb.gtBoardHexName()) {
        return false;
    }

    if (m_gtQuadInPsb != gtb.gtQuadInPsb()) {
        return false;
    }

    // all members identical
    return true;

}



// unequal operator
bool L1GtBoard::operator!=(const L1GtBoard& result) const
{

    return !( result == *this);

}

// less than operator
bool L1GtBoard::operator< (const L1GtBoard& gtb) const
{
    if (m_gtBoardType < gtb.gtBoardType()) {
        return true;
    } else {
        if (m_gtBoardType == gtb.gtBoardType()) {

            if (m_gtBoardIndex < gtb.gtBoardIndex()) {
                return true;
            }
        }
    }

    return false;
}

// set board type
void L1GtBoard::setGtBoardType(const L1GtBoardType& gtBoardTypeValue)
{
    m_gtBoardType = gtBoardTypeValue;
}

// set board index
void L1GtBoard::setGtBoardIndex(const int& gtBoardIndexValue)
{
    m_gtBoardIndex = gtBoardIndexValue;
}

// set the position of board data block
// in the GT DAQ readout record
void L1GtBoard::setGtPositionDaqRecord(const int& gtPositionDaqRecordValue)
{
    m_gtPositionDaqRecord = gtPositionDaqRecordValue;
}

// set the position of board data block
// in the GT EVM readout record
void L1GtBoard::setGtPositionEvmRecord(const int& gtPositionEvmRecordValue)
{
    m_gtPositionEvmRecord = gtPositionEvmRecordValue;
}

// set the bit of board in the GTFE ACTIVE_BOARDS
// for the GT DAQ readout record
void L1GtBoard::setGtBitDaqActiveBoards(const int& gtBitDaqActiveBoardsValue)
{
    m_gtBitDaqActiveBoards = gtBitDaqActiveBoardsValue;
}


// set the bit of board in the GTFE ACTIVE_BOARDS
// for the GT EVM readout record
void L1GtBoard::setGtBitEvmActiveBoards(const int& gtBitEvmActiveBoardsValue)
{
    m_gtBitEvmActiveBoards = gtBitEvmActiveBoardsValue;
}

// set board slot
void L1GtBoard::setGtBoardSlot(const int& gtBoardSlotValue)
{
    m_gtBoardSlot = gtBoardSlotValue;
}

// set board hex fragment name in hw record
void L1GtBoard::setGtBoardHexName(const int& gtBoardHexNameValue)
{
    m_gtBoardHexName = gtBoardHexNameValue;
}


// set L1 quadruplet (4x16 bits)(cable) in the PSB input
// valid for PSB only
void L1GtBoard::setGtQuadInPsb(const std::vector<L1GtPsbQuad>& gtQuadInPsbValue)
{
    m_gtQuadInPsb = gtQuadInPsbValue;
}

// get the board ID
const boost::uint16_t L1GtBoard::gtBoardId() const
{

    boost::uint16_t boardIdValue = 0;

    if (m_gtBoardType == GTFE) {
        boardIdValue = boardIdValue | m_gtBoardSlot;
    } else {
        boardIdValue = boardIdValue | (m_gtBoardHexName << 8) | m_gtBoardSlot;
    }

    return boardIdValue;
}

// return board name - it depends on L1GtBoardType enum!!!
std::string L1GtBoard::gtBoardName() const
{

    std::string gtBoardNameValue;

    // active board, add its size
    switch (m_gtBoardType) {

        case GTFE: {
                gtBoardNameValue = "GTFE";
            }
            break;
        case FDL: {
                gtBoardNameValue = "FDL";
            }
            break;
        case PSB: {
                gtBoardNameValue = "PSB";
            }
            break;
        case GMT: {
                gtBoardNameValue = "GMT";
            }
            break;
        case TCS: {
                gtBoardNameValue = "TCS";
            }
            break;
        case TIM: {
                gtBoardNameValue = "TIM";
            }
            break;
        default: {

                // do nothing here
                // TODO throw exception instead of returning empty string?
            }
            break;
    }


    return gtBoardNameValue;

}

/// print board
void L1GtBoard::print(std::ostream& myCout) const
{

    boost::uint16_t boardId = gtBoardId();
    std::string boardName = gtBoardName();

    myCout
    << "Board ID:                        " << std::hex << boardId << std::dec << std::endl
    << "Board Name:                      " << boardName << "_" << m_gtBoardIndex << std::endl
    << "Position in DAQ Record:          " << m_gtPositionDaqRecord << std::endl
    << "Position in EVM Record:          " << m_gtPositionEvmRecord << std::endl
    << "Active_Boards bit in DAQ Record: " << m_gtBitDaqActiveBoards << std::endl
    << "Active_Boards bit in EVM Record: " << m_gtBitEvmActiveBoards << std::endl
    << "Board HexName:                   " << std::hex << m_gtBoardHexName << std::dec
    << std::endl;

    if (m_gtBoardType == PSB) {
        myCout
        << "PSB Input per Cable: "
        << std::endl;
    }


    for (std::vector<L1GtPsbQuad>::const_iterator
            cIt = m_gtQuadInPsb.begin(); cIt != m_gtQuadInPsb.end(); ++cIt) {

        std::string objType;

        if ( *cIt == TechTr ) {
            objType = "TechTr";
        } else if ( *cIt == IsoEGQ ) {
            objType = "IsoEGQ";
        } else if ( *cIt == NoIsoEGQ ) {
            objType = "NoIsoEGQ";
        } else if ( *cIt == CenJetQ ) {
            objType = "CenJetQ";
        } else if ( *cIt == ForJetQ ) {
            objType = "ForJetQ";
        } else if ( *cIt == TauJetQ ) {
            objType = "TauJetQ";
        } else if ( *cIt == ESumsQ ) {
            objType = "ESumsQ";
        } else if ( *cIt == JetCountsQ ) {
            objType = "JetCountsQ";
        } else if ( *cIt == MQB1 ) {
            objType = "MQB1";
        } else if ( *cIt == MQB2 ) {
            objType = "MQB2";
        } else if ( *cIt == MQF3 ) {
            objType = "MQF3";
        } else if ( *cIt == MQF4 ) {
            objType = "MQF4";
        } else if ( *cIt == MQB5 ) {
            objType = "MQB5";
        } else if ( *cIt == MQB6 ) {
            objType = "MQB6";
        } else if ( *cIt == MQF7 ) {
            objType = "MQF7";
        } else if ( *cIt == MQF8 ) {
            objType = "MQF8";
        } else if ( *cIt == MQB9 ) {
            objType = "MQB9";
        } else if ( *cIt == MQB10 ) {
            objType = "MQB10";
        } else if ( *cIt == MQF11 ) {
            objType = "MQF11";
        } else if ( *cIt == MQF12 ) {
            objType = "MQF12";
        } else if ( *cIt == Free ) {
            objType = "Free";
        } else {
            // do nothing, return empty string
        }


        myCout << "       " << objType << " ";
    }

    myCout << std::endl;
}

// number of InfiniCables per board
const int L1GtBoard::NumberCablesBoard = 4;
