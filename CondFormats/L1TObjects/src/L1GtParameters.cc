/**
 * \class L1GtParameters
 *
 *
 * Description: L1 GT parameters.
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
#include "CondFormats/L1TObjects/interface/L1GtParameters.h"

// system include files
#include <iomanip>


// user include files
//   base class

// forward declarations

// constructor
L1GtParameters::L1GtParameters()
{
    // empty
}

// destructor
L1GtParameters::~L1GtParameters()
{
    // empty
}

// set the total Bx's in the event
void L1GtParameters::setGtTotalBxInEvent(const int& totalBxInEventValue)
{

    m_totalBxInEvent = totalBxInEventValue;

}

// set active boards for L1 GT DAQ record
void L1GtParameters::setGtDaqActiveBoards(const boost::uint16_t& activeBoardsValue)
{

    m_daqActiveBoards = activeBoardsValue;

}

// set active boards for L1 GT EVM record
void L1GtParameters::setGtEvmActiveBoards(const boost::uint16_t& activeBoardsValue)
{

    m_evmActiveBoards = activeBoardsValue;

}

// set number of Bx for L1 GT DAQ record
void L1GtParameters::setGtDaqNrBxBoard(const std::vector<int>& nrBxBoardValue)
{

    m_daqNrBxBoard = nrBxBoardValue;

}

// set number of Bx for L1 GT EVM record
void L1GtParameters::setGtEvmNrBxBoard(const std::vector<int>& nrBxBoardValue)
{

    m_evmNrBxBoard = nrBxBoardValue;

}

void L1GtParameters::setGtBstLengthBytes(const unsigned int& bstLengthBytesValue) {

    m_bstLengthBytes = bstLengthBytesValue;

}

// print all the L1 GT parameters
void L1GtParameters::print(std::ostream& myCout) const
{
    myCout
    << "\nL1 GT Parameters" << std::endl;

    myCout
    << "\n  Total Bx's in the event             = " << m_totalBxInEvent << std::endl;

    myCout
    << "\n  Active boards in L1 GT DAQ record (hex format) = "
    << std::hex << std::setw(sizeof(m_daqActiveBoards)*2) << std::setfill('0')
    << m_daqActiveBoards
    << std::dec << std::setfill(' ')
    << std::endl;

    myCout
    << "\n  Active boards in L1 GT EVM record (hex format) = "
    << std::hex << std::setw(sizeof(m_evmActiveBoards)*2) << std::setfill('0')
    << m_evmActiveBoards
    << std::dec << std::setfill(' ')
    << std::endl;

    myCout
    << "\n"
    << "  Number of bunch crosses per board in L1 GT DAQ record\n"
    << "  Board bit gives the position of the board in the \"active boards\" word.\n"
    << std::endl;

    int iBit = 0;
    for (int cIt : m_daqNrBxBoard) {
        myCout << "    Board active bit " << iBit << ": " << cIt << " BXs" << std::endl;
        iBit++;
    }


    myCout
    << "\n"
    << "  Number of bunch crosses per board in L1 GT EVM record\n"
    << "  Board bit gives the position of the board in the \"active boards\" word.\n"
    << std::endl;

    iBit = 0;
    for (int cIt : m_evmNrBxBoard) {
        myCout << "    Board active bit " << iBit << ": " << cIt << " BXs" << std::endl;
        iBit++;
    }

    myCout
    << "\n  Length of the BST message = "
    << m_bstLengthBytes
    << std::endl;

    myCout
    << "\n" << std::endl;

}

// output stream operator
std::ostream& operator<<(std::ostream& os, const L1GtParameters& result)
{
    result.print(os);
    return os;

}
