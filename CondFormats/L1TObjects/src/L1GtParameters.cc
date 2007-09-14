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
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "CondFormats/L1TObjects/interface/L1GtParameters.h"

// system include files
#include <ostream>
#include <iomanip>

#include <boost/cstdint.hpp>

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
void L1GtParameters::setGtTotalBxInEvent(int& totalBxInEventValue)
{

    m_totalBxInEvent = totalBxInEventValue;

}

// set the active boards
void L1GtParameters::setGtActiveBoards(boost::uint16_t& activeBoardsValue)
{

    m_activeBoards = activeBoardsValue;

}

// print all the L1 GT parameters
void L1GtParameters::print(std::ostream& s) const
{
    s << "\nL1 GT Parameters" << std::endl;

    s << "\n  Total Bx's in the event             = " << m_totalBxInEvent << std::endl;

    s << "\n  Active boards in L1 GT (hex format) = "
    << std::hex << std::setw(sizeof(m_activeBoards)*2) << std::setfill('0')
    << m_activeBoards
    << std::dec << std::setfill(' ')
    << std::endl;

    s << "\n" << std::endl;

}
