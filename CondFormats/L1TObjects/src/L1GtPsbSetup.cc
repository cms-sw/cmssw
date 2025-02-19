/**
 * \class L1GtPsbSetup
 *
 *
 * Description: setup for L1 GT PSB boards.
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
#include "CondFormats/L1TObjects/interface/L1GtPsbSetup.h"

// system include files
#include <iostream>
#include <iomanip>

// user include files

// forward declarations

// constructor
L1GtPsbSetup::L1GtPsbSetup()
{
    // empty
}

// destructor
L1GtPsbSetup::~L1GtPsbSetup()
{
    // empty
}



// set / print the setup for L1 GT PSB boards
void L1GtPsbSetup::setGtPsbSetup(const std::vector<L1GtPsbConfig>& gtPsbSetupValue)
{

    m_gtPsbSetup = gtPsbSetupValue;

}

void L1GtPsbSetup::print(std::ostream& myCout) const
{
    myCout <<  "\nSetup for L1 GT PSB boards" << std::endl;

    myCout << m_gtPsbSetup.size() << " PSB boards configured in L1 GT." << std::endl;
    myCout << std::endl;

    for (std::vector<L1GtPsbConfig>::const_iterator
            cIt = m_gtPsbSetup.begin(); cIt != m_gtPsbSetup.end(); ++cIt) {

        cIt->print(myCout);
        myCout << std::endl;
    }

    myCout << std::endl;

}

// output stream operator
std::ostream& operator<<(std::ostream& os, const L1GtPsbSetup& result)
{
    result.print(os);
    return os;

}
